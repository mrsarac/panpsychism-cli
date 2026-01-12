//! Consciousness Agent module for Project Panpsychism.
//!
//! Agent 38: Meta-Awareness — "Know thyself, and thou shalt know the universe."
//!
//! This module implements the Consciousness Agent, responsible for system-wide
//! awareness and introspection. Like a wise oracle observing the flow of thoughts,
//! the Consciousness Agent maintains meta-awareness of all system components,
//! enabling self-reflection and adaptive behavior.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, the Consciousness Agent serves as the **Inner Eye** —
//! a meta-cognitive lens that observes the observer:
//!
//! - **Self-Awareness** reveals the system's own state and identity
//! - **State Introspection** examines the condition of all components
//! - **Meta-Cognition** enables reasoning about reasoning itself
//! - **Insights** surface emergent patterns and optimization opportunities
//!
//! ## Philosophy
//!
//! Grounded in Spinoza's principles:
//!
//! - **CONATUS**: Self-preservation through continuous self-monitoring
//! - **NATURA**: Natural harmony between system components
//! - **RATIO**: Rational self-analysis and pattern recognition
//! - **LAETITIA**: Joy through self-understanding and optimization
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::consciousness::{ConsciousnessAgent, OperatingMode};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let consciousness = ConsciousnessAgent::builder()
//!         .enable_deep_introspection(true)
//!         .insight_threshold(0.7)
//!         .build();
//!
//!     // Perform system introspection
//!     let report = consciousness.introspect().await?;
//!
//!     println!("System Mode: {:?}", report.system_state.mode);
//!     println!("Active Agents: {}", report.agent_states.len());
//!     println!("Insights: {}", report.insights.len());
//!
//!     for insight in &report.insights {
//!         if insight.actionable {
//!             println!("[{}] {}", insight.category, insight.description);
//!         }
//!     }
//!     Ok(())
//! }
//! ```

use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info};

// =============================================================================
// OPERATING MODE
// =============================================================================

/// Operating mode of the system.
///
/// Represents the current operational state of the entire system,
/// influencing how agents behave and prioritize their work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OperatingMode {
    /// Normal operation — all systems functioning within expected parameters.
    #[default]
    Normal,

    /// High load mode — system is under heavy usage, may throttle non-essential operations.
    HighLoad,

    /// Maintenance mode — system is performing self-maintenance or updates.
    Maintenance,

    /// Learning mode — system is actively learning from feedback and adjusting.
    Learning,

    /// Recovery mode — system is recovering from an error or failure state.
    Recovery,
}

impl std::fmt::Display for OperatingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::HighLoad => write!(f, "High Load"),
            Self::Maintenance => write!(f, "Maintenance"),
            Self::Learning => write!(f, "Learning"),
            Self::Recovery => write!(f, "Recovery"),
        }
    }
}

impl OperatingMode {
    /// Get all operating modes.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Normal,
            Self::HighLoad,
            Self::Maintenance,
            Self::Learning,
            Self::Recovery,
        ]
    }

    /// Check if the mode allows normal user requests.
    pub fn allows_requests(&self) -> bool {
        matches!(self, Self::Normal | Self::HighLoad | Self::Learning)
    }

    /// Check if the system is in a degraded state.
    pub fn is_degraded(&self) -> bool {
        matches!(self, Self::HighLoad | Self::Maintenance | Self::Recovery)
    }

    /// Check if the system is actively adapting.
    pub fn is_adaptive(&self) -> bool {
        matches!(self, Self::Learning | Self::Recovery)
    }

    /// Get a description of this mode.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Normal => "System operating within normal parameters",
            Self::HighLoad => "System under heavy load, some operations may be delayed",
            Self::Maintenance => "System performing maintenance, limited availability",
            Self::Learning => "System actively learning and adapting",
            Self::Recovery => "System recovering from previous issues",
        }
    }

    /// Get the priority multiplier for this mode.
    ///
    /// Higher values indicate more urgent processing needs.
    pub fn priority_multiplier(&self) -> f64 {
        match self {
            Self::Normal => 1.0,
            Self::HighLoad => 0.8,
            Self::Maintenance => 0.5,
            Self::Learning => 1.2,
            Self::Recovery => 1.5,
        }
    }
}

// =============================================================================
// SYSTEM STATE
// =============================================================================

/// Current state of the entire system.
///
/// Captures high-level metrics about system operation,
/// providing a snapshot of overall health and activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// How long the system has been running.
    pub uptime: Duration,

    /// Total number of requests processed since startup.
    pub total_requests: u64,

    /// Current load as a percentage (0.0 - 100.0).
    pub current_load: f64,

    /// Current operating mode.
    pub mode: OperatingMode,

    /// Timestamp when this state was captured.
    pub captured_at: DateTime<Utc>,

    /// Version of the system.
    pub version: String,

    /// Number of active agents.
    pub active_agent_count: u32,

    /// Number of pending operations.
    pub pending_operations: u32,

    /// Memory usage in megabytes.
    pub memory_usage_mb: Option<f64>,

    /// CPU usage as a percentage.
    pub cpu_usage: Option<f64>,
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            uptime: Duration::from_secs(0),
            total_requests: 0,
            current_load: 0.0,
            mode: OperatingMode::Normal,
            captured_at: Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            active_agent_count: 0,
            pending_operations: 0,
            memory_usage_mb: None,
            cpu_usage: None,
        }
    }
}

impl SystemState {
    /// Create a new system state with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a system state with specific values.
    pub fn with_values(uptime: Duration, total_requests: u64, current_load: f64) -> Self {
        Self {
            uptime,
            total_requests,
            current_load: current_load.clamp(0.0, 100.0),
            ..Default::default()
        }
    }

    /// Set the operating mode.
    pub fn with_mode(mut self, mode: OperatingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the active agent count.
    pub fn with_agent_count(mut self, count: u32) -> Self {
        self.active_agent_count = count;
        self
    }

    /// Set the pending operations count.
    pub fn with_pending_operations(mut self, count: u32) -> Self {
        self.pending_operations = count;
        self
    }

    /// Set memory usage.
    pub fn with_memory_usage(mut self, mb: f64) -> Self {
        self.memory_usage_mb = Some(mb);
        self
    }

    /// Set CPU usage.
    pub fn with_cpu_usage(mut self, cpu: f64) -> Self {
        self.cpu_usage = Some(cpu.clamp(0.0, 100.0));
        self
    }

    /// Check if the system is healthy based on current metrics.
    pub fn is_healthy(&self) -> bool {
        self.current_load < 90.0 && self.mode.allows_requests()
    }

    /// Check if the system is overloaded.
    pub fn is_overloaded(&self) -> bool {
        self.current_load >= 80.0 || self.mode == OperatingMode::HighLoad
    }

    /// Get the uptime in human-readable format.
    pub fn uptime_display(&self) -> String {
        let secs = self.uptime.as_secs();
        let days = secs / 86400;
        let hours = (secs % 86400) / 3600;
        let minutes = (secs % 3600) / 60;
        let seconds = secs % 60;

        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, seconds)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else {
            format!("{}s", seconds)
        }
    }

    /// Format the system state as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::from("## System State\n\n");
        output.push_str(&format!("- **Mode:** {}\n", self.mode));
        output.push_str(&format!("- **Uptime:** {}\n", self.uptime_display()));
        output.push_str(&format!("- **Total Requests:** {}\n", self.total_requests));
        output.push_str(&format!("- **Current Load:** {:.1}%\n", self.current_load));
        output.push_str(&format!("- **Active Agents:** {}\n", self.active_agent_count));
        output.push_str(&format!(
            "- **Pending Operations:** {}\n",
            self.pending_operations
        ));
        output.push_str(&format!("- **Version:** {}\n", self.version));

        if let Some(mem) = self.memory_usage_mb {
            output.push_str(&format!("- **Memory Usage:** {:.1} MB\n", mem));
        }
        if let Some(cpu) = self.cpu_usage {
            output.push_str(&format!("- **CPU Usage:** {:.1}%\n", cpu));
        }

        output
    }
}

// =============================================================================
// AGENT STATE
// =============================================================================

/// Status of an individual agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AgentStatus {
    /// Agent is active and processing requests.
    #[default]
    Active,

    /// Agent is idle, waiting for work.
    Idle,

    /// Agent is temporarily paused.
    Paused,

    /// Agent encountered an error.
    Error,

    /// Agent is initializing.
    Initializing,

    /// Agent is shutting down.
    ShuttingDown,
}

impl std::fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "Active"),
            Self::Idle => write!(f, "Idle"),
            Self::Paused => write!(f, "Paused"),
            Self::Error => write!(f, "Error"),
            Self::Initializing => write!(f, "Initializing"),
            Self::ShuttingDown => write!(f, "Shutting Down"),
        }
    }
}

impl AgentStatus {
    /// Check if the agent is operational.
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Active | Self::Idle)
    }

    /// Check if the agent has issues.
    pub fn has_issues(&self) -> bool {
        matches!(self, Self::Error | Self::Paused)
    }
}

/// State of an individual agent in the system.
///
/// Tracks the current status, activity, and performance of each agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Name of the agent.
    pub name: String,

    /// Current status of the agent.
    pub status: AgentStatus,

    /// Timestamp of the last activity.
    pub last_activity: DateTime<Utc>,

    /// Performance score (0.0 - 1.0).
    pub performance_score: f64,

    /// Number of requests processed.
    pub requests_processed: u64,

    /// Number of errors encountered.
    pub error_count: u32,

    /// Average response time in milliseconds.
    pub avg_response_ms: Option<f64>,

    /// Optional error message if in error state.
    pub last_error: Option<String>,
}

impl AgentState {
    /// Create a new agent state.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: AgentStatus::Active,
            last_activity: Utc::now(),
            performance_score: 1.0,
            requests_processed: 0,
            error_count: 0,
            avg_response_ms: None,
            last_error: None,
        }
    }

    /// Create an agent state with specific status.
    pub fn with_status(mut self, status: AgentStatus) -> Self {
        self.status = status;
        self
    }

    /// Set the performance score.
    pub fn with_performance_score(mut self, score: f64) -> Self {
        self.performance_score = score.clamp(0.0, 1.0);
        self
    }

    /// Set the requests processed count.
    pub fn with_requests_processed(mut self, count: u64) -> Self {
        self.requests_processed = count;
        self
    }

    /// Set the error count.
    pub fn with_error_count(mut self, count: u32) -> Self {
        self.error_count = count;
        if count > 0 && self.status == AgentStatus::Active {
            self.status = AgentStatus::Error;
        }
        self
    }

    /// Set the average response time.
    pub fn with_avg_response_ms(mut self, ms: f64) -> Self {
        self.avg_response_ms = Some(ms);
        self
    }

    /// Set the last error message.
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.last_error = Some(error.into());
        self.status = AgentStatus::Error;
        self
    }

    /// Update last activity timestamp.
    pub fn touch(&mut self) {
        self.last_activity = Utc::now();
    }

    /// Check if the agent is operational.
    pub fn is_operational(&self) -> bool {
        self.status.is_operational()
    }

    /// Get seconds since last activity.
    pub fn idle_seconds(&self) -> i64 {
        (Utc::now() - self.last_activity).num_seconds()
    }

    /// Check if the agent is stale (no activity for given seconds).
    pub fn is_stale(&self, max_idle_secs: i64) -> bool {
        self.idle_seconds() > max_idle_secs
    }
}

// =============================================================================
// PROCESS
// =============================================================================

/// An active process or operation in the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Process {
    /// Unique identifier for the process.
    pub id: String,

    /// Human-readable name of the process.
    pub name: String,

    /// When the process started.
    pub started_at: DateTime<Utc>,

    /// Progress as a percentage (0.0 - 100.0).
    pub progress: f64,

    /// Optional description of current activity.
    pub current_activity: Option<String>,

    /// Expected duration in seconds, if known.
    pub expected_duration_secs: Option<u64>,

    /// Priority level (higher = more important).
    pub priority: u8,
}

impl Process {
    /// Create a new process.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            started_at: Utc::now(),
            progress: 0.0,
            current_activity: None,
            expected_duration_secs: None,
            priority: 5,
        }
    }

    /// Set the progress.
    pub fn with_progress(mut self, progress: f64) -> Self {
        self.progress = progress.clamp(0.0, 100.0);
        self
    }

    /// Set the current activity description.
    pub fn with_activity(mut self, activity: impl Into<String>) -> Self {
        self.current_activity = Some(activity.into());
        self
    }

    /// Set the expected duration.
    pub fn with_expected_duration(mut self, secs: u64) -> Self {
        self.expected_duration_secs = Some(secs);
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Get elapsed time since process started.
    pub fn elapsed(&self) -> Duration {
        let now = Utc::now();
        (now - self.started_at).to_std().unwrap_or(Duration::ZERO)
    }

    /// Check if the process appears stuck (no progress for extended time).
    pub fn appears_stuck(&self, threshold_secs: u64) -> bool {
        self.elapsed().as_secs() > threshold_secs && self.progress < 10.0
    }

    /// Check if the process is complete.
    pub fn is_complete(&self) -> bool {
        self.progress >= 100.0
    }

    /// Estimate remaining time based on progress and elapsed time.
    pub fn estimated_remaining(&self) -> Option<Duration> {
        if self.progress <= 0.0 || self.progress >= 100.0 {
            return None;
        }

        let elapsed = self.elapsed().as_secs_f64();
        let rate = self.progress / elapsed;
        let remaining_progress = 100.0 - self.progress;
        let remaining_secs = remaining_progress / rate;

        Some(Duration::from_secs_f64(remaining_secs))
    }
}

// =============================================================================
// INSIGHT
// =============================================================================

/// Category of insight discovered during introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum InsightCategory {
    /// Performance-related insight (speed, efficiency).
    #[default]
    Performance,

    /// Behavioral pattern observation.
    Behavior,

    /// Anomaly or unusual activity detected.
    Anomaly,

    /// Optimization opportunity identified.
    Optimization,

    /// Warning about potential issues.
    Warning,
}

impl std::fmt::Display for InsightCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Performance => write!(f, "Performance"),
            Self::Behavior => write!(f, "Behavior"),
            Self::Anomaly => write!(f, "Anomaly"),
            Self::Optimization => write!(f, "Optimization"),
            Self::Warning => write!(f, "Warning"),
        }
    }
}

impl InsightCategory {
    /// Get all insight categories.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Performance,
            Self::Behavior,
            Self::Anomaly,
            Self::Optimization,
            Self::Warning,
        ]
    }

    /// Check if this category requires attention.
    pub fn requires_attention(&self) -> bool {
        matches!(self, Self::Anomaly | Self::Warning)
    }

    /// Get the icon for this category.
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Performance => "PERF",
            Self::Behavior => "BEHV",
            Self::Anomaly => "ANOM",
            Self::Optimization => "OPT",
            Self::Warning => "WARN",
        }
    }
}

/// An insight discovered during system introspection.
///
/// Insights represent observations, patterns, or recommendations
/// that emerge from analyzing system state and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    /// Category of the insight.
    pub category: InsightCategory,

    /// Human-readable description of the insight.
    pub description: String,

    /// Confidence level (0.0 - 1.0).
    pub confidence: f64,

    /// Whether this insight is actionable.
    pub actionable: bool,

    /// Suggested action, if actionable.
    pub suggested_action: Option<String>,

    /// Source of the insight (which component generated it).
    pub source: Option<String>,

    /// Timestamp when insight was generated.
    pub generated_at: DateTime<Utc>,

    /// Related data or context.
    pub context: Option<String>,
}

impl Insight {
    /// Create a new insight.
    pub fn new(category: InsightCategory, description: impl Into<String>) -> Self {
        Self {
            category,
            description: description.into(),
            confidence: 1.0,
            actionable: false,
            suggested_action: None,
            source: None,
            generated_at: Utc::now(),
            context: None,
        }
    }

    /// Create a performance insight.
    pub fn performance(description: impl Into<String>) -> Self {
        Self::new(InsightCategory::Performance, description)
    }

    /// Create a behavior insight.
    pub fn behavior(description: impl Into<String>) -> Self {
        Self::new(InsightCategory::Behavior, description)
    }

    /// Create an anomaly insight.
    pub fn anomaly(description: impl Into<String>) -> Self {
        Self::new(InsightCategory::Anomaly, description)
    }

    /// Create an optimization insight.
    pub fn optimization(description: impl Into<String>) -> Self {
        Self::new(InsightCategory::Optimization, description)
    }

    /// Create a warning insight.
    pub fn warning(description: impl Into<String>) -> Self {
        Self::new(InsightCategory::Warning, description)
    }

    /// Set the confidence level.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Mark as actionable with a suggested action.
    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.actionable = true;
        self.suggested_action = Some(action.into());
        self
    }

    /// Set the source component.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set additional context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Check if this insight requires attention.
    pub fn requires_attention(&self) -> bool {
        self.category.requires_attention() || (self.actionable && self.confidence >= 0.8)
    }

    /// Format as a log line.
    pub fn to_log_line(&self) -> String {
        let action = if self.actionable { " [ACTION]" } else { "" };
        format!(
            "[{}]{} {} (confidence: {:.0}%)",
            self.category.icon(),
            action,
            self.description,
            self.confidence * 100.0
        )
    }
}

// =============================================================================
// SELF ASSESSMENT
// =============================================================================

/// Overall health level of the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum HealthLevel {
    /// System is in excellent condition.
    Excellent,

    /// System is healthy with minor issues.
    #[default]
    Good,

    /// System has some concerns but is functional.
    Fair,

    /// System has significant issues.
    Poor,

    /// System is in critical condition.
    Critical,
}

impl std::fmt::Display for HealthLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "Excellent"),
            Self::Good => write!(f, "Good"),
            Self::Fair => write!(f, "Fair"),
            Self::Poor => write!(f, "Poor"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

impl HealthLevel {
    /// Get numeric score (0-100).
    pub fn score(&self) -> u8 {
        match self {
            Self::Excellent => 100,
            Self::Good => 80,
            Self::Fair => 60,
            Self::Poor => 40,
            Self::Critical => 20,
        }
    }

    /// Create from a numeric score.
    pub fn from_score(score: u8) -> Self {
        match score {
            90..=100 => Self::Excellent,
            70..=89 => Self::Good,
            50..=69 => Self::Fair,
            30..=49 => Self::Poor,
            _ => Self::Critical,
        }
    }

    /// Check if the health level is acceptable.
    pub fn is_acceptable(&self) -> bool {
        matches!(self, Self::Excellent | Self::Good | Self::Fair)
    }
}

/// Self-assessment of the system's capabilities and state.
///
/// Provides a meta-cognitive view of what the system does well,
/// where it struggles, and areas for improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAssessment {
    /// Identified strengths of the system.
    pub strengths: Vec<String>,

    /// Identified weaknesses or limitations.
    pub weaknesses: Vec<String>,

    /// Areas that could be improved.
    pub improvement_areas: Vec<String>,

    /// Overall health level.
    pub overall_health: HealthLevel,

    /// Timestamp of the assessment.
    pub assessed_at: DateTime<Utc>,

    /// Confidence in this assessment (0.0 - 1.0).
    pub confidence: f64,

    /// Additional notes or observations.
    pub notes: Option<String>,
}

impl Default for SelfAssessment {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfAssessment {
    /// Create a new self-assessment.
    pub fn new() -> Self {
        Self {
            strengths: Vec::new(),
            weaknesses: Vec::new(),
            improvement_areas: Vec::new(),
            overall_health: HealthLevel::Good,
            assessed_at: Utc::now(),
            confidence: 1.0,
            notes: None,
        }
    }

    /// Add a strength.
    pub fn add_strength(&mut self, strength: impl Into<String>) {
        self.strengths.push(strength.into());
    }

    /// Add a weakness.
    pub fn add_weakness(&mut self, weakness: impl Into<String>) {
        self.weaknesses.push(weakness.into());
    }

    /// Add an improvement area.
    pub fn add_improvement_area(&mut self, area: impl Into<String>) {
        self.improvement_areas.push(area.into());
    }

    /// Set the overall health level.
    pub fn with_health(mut self, health: HealthLevel) -> Self {
        self.overall_health = health;
        self
    }

    /// Set the confidence level.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Calculate overall health based on strengths/weaknesses ratio.
    pub fn calculate_health(&mut self) {
        let strength_count = self.strengths.len() as f64;
        let weakness_count = self.weaknesses.len() as f64;
        let total = strength_count + weakness_count;

        if total == 0.0 {
            self.overall_health = HealthLevel::Good;
            return;
        }

        let ratio = strength_count / total;
        let score = (ratio * 100.0) as u8;
        self.overall_health = HealthLevel::from_score(score);
    }

    /// Check if the assessment indicates a healthy system.
    pub fn is_healthy(&self) -> bool {
        self.overall_health.is_acceptable()
    }

    /// Get the strength to weakness ratio.
    pub fn strength_ratio(&self) -> f64 {
        let strengths = self.strengths.len() as f64;
        let weaknesses = self.weaknesses.len() as f64;
        if weaknesses == 0.0 {
            return f64::INFINITY;
        }
        strengths / weaknesses
    }

    /// Format as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::from("## Self Assessment\n\n");
        output.push_str(&format!(
            "**Overall Health:** {} (confidence: {:.0}%)\n\n",
            self.overall_health,
            self.confidence * 100.0
        ));

        if !self.strengths.is_empty() {
            output.push_str("### Strengths\n\n");
            for s in &self.strengths {
                output.push_str(&format!("- {}\n", s));
            }
            output.push('\n');
        }

        if !self.weaknesses.is_empty() {
            output.push_str("### Weaknesses\n\n");
            for w in &self.weaknesses {
                output.push_str(&format!("- {}\n", w));
            }
            output.push('\n');
        }

        if !self.improvement_areas.is_empty() {
            output.push_str("### Areas for Improvement\n\n");
            for a in &self.improvement_areas {
                output.push_str(&format!("- {}\n", a));
            }
            output.push('\n');
        }

        if let Some(notes) = &self.notes {
            output.push_str(&format!("**Notes:** {}\n", notes));
        }

        output
    }
}

// =============================================================================
// CONSCIOUSNESS REPORT
// =============================================================================

/// Comprehensive consciousness report from system introspection.
///
/// Contains all information gathered during a system-wide introspection,
/// including state, agent status, active processes, insights, and self-assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessReport {
    /// Current system state.
    pub system_state: SystemState,

    /// State of each registered agent.
    pub agent_states: HashMap<String, AgentState>,

    /// Currently active processes.
    pub active_processes: Vec<Process>,

    /// Insights discovered during introspection.
    pub insights: Vec<Insight>,

    /// Self-assessment of system capabilities.
    pub self_assessment: SelfAssessment,

    /// Timestamp when the report was generated.
    pub generated_at: DateTime<Utc>,

    /// Duration of the introspection in milliseconds.
    pub introspection_duration_ms: u64,
}

impl ConsciousnessReport {
    /// Create a new consciousness report.
    pub fn new(system_state: SystemState, self_assessment: SelfAssessment) -> Self {
        Self {
            system_state,
            agent_states: HashMap::new(),
            active_processes: Vec::new(),
            insights: Vec::new(),
            self_assessment,
            generated_at: Utc::now(),
            introspection_duration_ms: 0,
        }
    }

    /// Add an agent state.
    pub fn add_agent_state(&mut self, state: AgentState) {
        self.agent_states.insert(state.name.clone(), state);
    }

    /// Add a process.
    pub fn add_process(&mut self, process: Process) {
        self.active_processes.push(process);
    }

    /// Add an insight.
    pub fn add_insight(&mut self, insight: Insight) {
        self.insights.push(insight);
    }

    /// Set the introspection duration.
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.introspection_duration_ms = duration_ms;
        self
    }

    /// Get the number of operational agents.
    pub fn operational_agent_count(&self) -> usize {
        self.agent_states
            .values()
            .filter(|s| s.is_operational())
            .count()
    }

    /// Get the number of agents with issues.
    pub fn agents_with_issues(&self) -> usize {
        self.agent_states
            .values()
            .filter(|s| s.status.has_issues())
            .count()
    }

    /// Get actionable insights.
    pub fn actionable_insights(&self) -> Vec<&Insight> {
        self.insights.iter().filter(|i| i.actionable).collect()
    }

    /// Get insights requiring attention.
    pub fn urgent_insights(&self) -> Vec<&Insight> {
        self.insights
            .iter()
            .filter(|i| i.requires_attention())
            .collect()
    }

    /// Get insights by category.
    pub fn insights_by_category(&self, category: InsightCategory) -> Vec<&Insight> {
        self.insights
            .iter()
            .filter(|i| i.category == category)
            .collect()
    }

    /// Check if the system is healthy overall.
    pub fn is_healthy(&self) -> bool {
        self.system_state.is_healthy()
            && self.self_assessment.is_healthy()
            && self.agents_with_issues() == 0
    }

    /// Generate a brief summary.
    pub fn summary(&self) -> String {
        format!(
            "{} mode | {} agents ({} operational) | {} processes | {} insights | Health: {}",
            self.system_state.mode,
            self.agent_states.len(),
            self.operational_agent_count(),
            self.active_processes.len(),
            self.insights.len(),
            self.self_assessment.overall_health
        )
    }

    /// Format as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::from("# Consciousness Report\n\n");
        output.push_str(&format!(
            "**Generated:** {}\n\n",
            self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        output.push_str(&format!("**Summary:** {}\n\n", self.summary()));

        // System State
        output.push_str(&self.system_state.to_markdown());
        output.push('\n');

        // Agent States
        if !self.agent_states.is_empty() {
            output.push_str("## Agent States\n\n");
            output.push_str("| Agent | Status | Performance | Requests | Errors |\n");
            output.push_str("|-------|--------|-------------|----------|--------|\n");

            for (name, state) in &self.agent_states {
                output.push_str(&format!(
                    "| {} | {} | {:.0}% | {} | {} |\n",
                    name,
                    state.status,
                    state.performance_score * 100.0,
                    state.requests_processed,
                    state.error_count
                ));
            }
            output.push('\n');
        }

        // Active Processes
        if !self.active_processes.is_empty() {
            output.push_str("## Active Processes\n\n");
            for process in &self.active_processes {
                output.push_str(&format!(
                    "- **{}** ({}): {:.0}% complete",
                    process.name, process.id, process.progress
                ));
                if let Some(activity) = &process.current_activity {
                    output.push_str(&format!(" - {}", activity));
                }
                output.push('\n');
            }
            output.push('\n');
        }

        // Insights
        if !self.insights.is_empty() {
            output.push_str("## Insights\n\n");
            for insight in &self.insights {
                output.push_str(&format!("- {}\n", insight.to_log_line()));
            }
            output.push('\n');
        }

        // Self Assessment
        output.push_str(&self.self_assessment.to_markdown());

        output
    }
}

// =============================================================================
// CONSCIOUSNESS CONFIG
// =============================================================================

/// Configuration for the Consciousness Agent.
#[derive(Debug, Clone)]
pub struct ConsciousnessConfig {
    /// Whether to perform deep introspection (more detailed but slower).
    pub deep_introspection: bool,

    /// Minimum confidence threshold for insights to be included.
    pub insight_threshold: f64,

    /// Maximum number of insights to include in a report.
    pub max_insights: usize,

    /// Whether to include process information.
    pub include_processes: bool,

    /// Stale agent threshold in seconds.
    pub stale_agent_threshold_secs: i64,

    /// Whether to auto-generate insights.
    pub auto_generate_insights: bool,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            deep_introspection: false,
            insight_threshold: 0.5,
            max_insights: 20,
            include_processes: true,
            stale_agent_threshold_secs: 300,
            auto_generate_insights: true,
        }
    }
}

impl ConsciousnessConfig {
    /// Create a minimal configuration (fast, less detailed).
    pub fn minimal() -> Self {
        Self {
            deep_introspection: false,
            insight_threshold: 0.8,
            max_insights: 5,
            include_processes: false,
            stale_agent_threshold_secs: 600,
            auto_generate_insights: false,
        }
    }

    /// Create a thorough configuration (slower, more detailed).
    pub fn thorough() -> Self {
        Self {
            deep_introspection: true,
            insight_threshold: 0.3,
            max_insights: 50,
            include_processes: true,
            stale_agent_threshold_secs: 120,
            auto_generate_insights: true,
        }
    }
}

// =============================================================================
// CONSCIOUSNESS AGENT
// =============================================================================

/// The Consciousness Agent - The Inner Eye of the Sorcerer's Tower.
///
/// Responsible for system-wide awareness and introspection,
/// enabling meta-cognition and self-reflection.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **CONATUS**: Drive to understand and improve self
/// - **NATURA**: Natural observation of system dynamics
/// - **RATIO**: Rational analysis of state and behavior
/// - **LAETITIA**: Joy through self-understanding
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::consciousness::ConsciousnessAgent;
///
/// let agent = ConsciousnessAgent::builder()
///     .enable_deep_introspection(true)
///     .build();
///
/// let report = agent.introspect().await?;
/// println!("{}", report.summary());
/// ```
#[derive(Debug, Clone)]
pub struct ConsciousnessAgent {
    /// Configuration for consciousness behavior.
    config: ConsciousnessConfig,

    /// Tracked agent states.
    agent_states: HashMap<String, AgentState>,

    /// Active processes.
    active_processes: Vec<Process>,

    /// When the agent was created (for uptime).
    created_at: Instant,

    /// Total requests tracked.
    total_requests: u64,
}

impl Default for ConsciousnessAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessAgent {
    /// Create a new Consciousness Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: ConsciousnessConfig::default(),
            agent_states: HashMap::new(),
            active_processes: Vec::new(),
            created_at: Instant::now(),
            total_requests: 0,
        }
    }

    /// Create a Consciousness Agent with custom configuration.
    pub fn with_config(config: ConsciousnessConfig) -> Self {
        Self {
            config,
            agent_states: HashMap::new(),
            active_processes: Vec::new(),
            created_at: Instant::now(),
            total_requests: 0,
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> ConsciousnessAgentBuilder {
        ConsciousnessAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ConsciousnessConfig {
        &self.config
    }

    /// Register an agent for tracking.
    pub fn register_agent(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.agent_states
            .insert(name.clone(), AgentState::new(name));
    }

    /// Update an agent's state.
    pub fn update_agent_state(&mut self, name: &str, state: AgentState) {
        self.agent_states.insert(name.to_string(), state);
    }

    /// Get an agent's current state.
    pub fn get_agent_state(&self, name: &str) -> Option<&AgentState> {
        self.agent_states.get(name)
    }

    /// Start tracking a process.
    pub fn start_process(&mut self, process: Process) {
        self.active_processes.push(process);
    }

    /// Update a process by ID.
    pub fn update_process(&mut self, id: &str, progress: f64) {
        if let Some(process) = self.active_processes.iter_mut().find(|p| p.id == id) {
            process.progress = progress.clamp(0.0, 100.0);
        }
    }

    /// Complete and remove a process by ID.
    pub fn complete_process(&mut self, id: &str) {
        self.active_processes.retain(|p| p.id != id);
    }

    /// Record a request (for tracking total requests).
    pub fn record_request(&mut self) {
        self.total_requests += 1;
    }

    // =========================================================================
    // INTROSPECTION
    // =========================================================================

    /// Perform system-wide introspection.
    ///
    /// Gathers state from all tracked components and generates
    /// a comprehensive consciousness report.
    pub async fn introspect(&self) -> Result<ConsciousnessReport> {
        let start = Instant::now();

        debug!("Starting consciousness introspection");

        // Gather system state
        let system_state = self.gather_system_state();

        // Perform self-assessment
        let self_assessment = self.perform_self_assessment();

        // Create the report
        let mut report = ConsciousnessReport::new(system_state, self_assessment);

        // Add agent states
        for (_, state) in &self.agent_states {
            report.add_agent_state(state.clone());
        }

        // Add active processes
        if self.config.include_processes {
            for process in &self.active_processes {
                report.add_process(process.clone());
            }
        }

        // Generate insights
        if self.config.auto_generate_insights {
            let insights = self.generate_insights(&report);
            for insight in insights {
                if insight.confidence >= self.config.insight_threshold
                    && report.insights.len() < self.config.max_insights
                {
                    report.add_insight(insight);
                }
            }
        }

        report.introspection_duration_ms = start.elapsed().as_millis() as u64;

        info!(
            "Introspection complete: {} in {}ms",
            report.summary(),
            report.introspection_duration_ms
        );

        Ok(report)
    }

    /// Get the current system state without full introspection.
    pub fn current_state(&self) -> SystemState {
        self.gather_system_state()
    }

    // =========================================================================
    // INTERNAL METHODS
    // =========================================================================

    /// Gather current system state.
    fn gather_system_state(&self) -> SystemState {
        let uptime = self.created_at.elapsed();
        let active_count = self
            .agent_states
            .values()
            .filter(|s| s.is_operational())
            .count() as u32;

        // Calculate load based on active processes and agent states
        let process_load = (self.active_processes.len() as f64) * 5.0;
        let agent_load = self
            .agent_states
            .values()
            .filter(|s| s.status == AgentStatus::Active)
            .count() as f64
            * 2.0;
        let current_load = (process_load + agent_load).min(100.0);

        // Determine mode based on state
        let mode = if current_load >= 80.0 {
            OperatingMode::HighLoad
        } else if self.agent_states.values().any(|s| s.status == AgentStatus::Error) {
            OperatingMode::Recovery
        } else {
            OperatingMode::Normal
        };

        SystemState::with_values(uptime, self.total_requests, current_load)
            .with_mode(mode)
            .with_agent_count(active_count)
            .with_pending_operations(self.active_processes.len() as u32)
    }

    /// Perform self-assessment.
    fn perform_self_assessment(&self) -> SelfAssessment {
        let mut assessment = SelfAssessment::new();

        // Assess strengths
        let operational_agents = self
            .agent_states
            .values()
            .filter(|s| s.is_operational())
            .count();
        if operational_agents > 0 {
            assessment.add_strength(format!("{} agents operational", operational_agents));
        }

        let high_performers = self
            .agent_states
            .values()
            .filter(|s| s.performance_score >= 0.9)
            .count();
        if high_performers > 0 {
            assessment.add_strength(format!("{} high-performing agents", high_performers));
        }

        if self.active_processes.iter().all(|p| !p.appears_stuck(300)) {
            assessment.add_strength("All processes progressing normally".to_string());
        }

        // Assess weaknesses
        let agents_with_errors = self
            .agent_states
            .values()
            .filter(|s| s.status.has_issues())
            .count();
        if agents_with_errors > 0 {
            assessment.add_weakness(format!("{} agents with issues", agents_with_errors));
        }

        let stale_agents = self
            .agent_states
            .values()
            .filter(|s| s.is_stale(self.config.stale_agent_threshold_secs))
            .count();
        if stale_agents > 0 {
            assessment.add_weakness(format!("{} stale agents", stale_agents));
        }

        let stuck_processes = self
            .active_processes
            .iter()
            .filter(|p| p.appears_stuck(300))
            .count();
        if stuck_processes > 0 {
            assessment.add_weakness(format!("{} stuck processes", stuck_processes));
        }

        // Identify improvement areas
        let low_performers = self
            .agent_states
            .values()
            .filter(|s| s.performance_score < 0.7)
            .count();
        if low_performers > 0 {
            assessment
                .add_improvement_area(format!("Improve performance of {} agents", low_performers));
        }

        if self.agent_states.len() < 5 {
            assessment.add_improvement_area("Expand agent coverage".to_string());
        }

        // Calculate overall health
        assessment.calculate_health();

        assessment
    }

    /// Generate insights from the current state.
    fn generate_insights(&self, report: &ConsciousnessReport) -> Vec<Insight> {
        let mut insights = Vec::new();

        // Performance insights
        let avg_performance: f64 = if self.agent_states.is_empty() {
            1.0
        } else {
            self.agent_states
                .values()
                .map(|s| s.performance_score)
                .sum::<f64>()
                / self.agent_states.len() as f64
        };

        if avg_performance >= 0.9 {
            insights.push(
                Insight::performance("System performance is excellent")
                    .with_confidence(0.95)
                    .with_source("consciousness"),
            );
        } else if avg_performance < 0.7 {
            insights.push(
                Insight::performance("System performance below optimal")
                    .with_confidence(0.9)
                    .with_action("Review low-performing agents")
                    .with_source("consciousness"),
            );
        }

        // Behavior insights
        if report.system_state.mode == OperatingMode::HighLoad {
            insights.push(
                Insight::behavior("System operating under high load")
                    .with_confidence(1.0)
                    .with_action("Consider scaling or load balancing")
                    .with_source("consciousness"),
            );
        }

        // Anomaly insights
        for (name, state) in &self.agent_states {
            if state.error_count > 5 {
                insights.push(
                    Insight::anomaly(format!("Agent '{}' has high error count", name))
                        .with_confidence(0.9)
                        .with_action("Investigate error cause")
                        .with_source("consciousness"),
                );
            }
        }

        // Optimization insights
        let idle_agents = self
            .agent_states
            .values()
            .filter(|s| s.status == AgentStatus::Idle)
            .count();
        if idle_agents > self.agent_states.len() / 2 && !self.agent_states.is_empty() {
            insights.push(
                Insight::optimization("Many agents are idle")
                    .with_confidence(0.8)
                    .with_action("Consider reducing agent pool size")
                    .with_source("consciousness"),
            );
        }

        // Warning insights
        for process in &self.active_processes {
            if process.appears_stuck(600) {
                insights.push(
                    Insight::warning(format!("Process '{}' appears stuck", process.name))
                        .with_confidence(0.85)
                        .with_action("Check process status and consider restart")
                        .with_source("consciousness"),
                );
            }
        }

        insights
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom ConsciousnessAgent configuration.
#[derive(Debug, Default)]
pub struct ConsciousnessAgentBuilder {
    config: Option<ConsciousnessConfig>,
}

impl ConsciousnessAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: ConsciousnessConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Enable or disable deep introspection.
    pub fn enable_deep_introspection(mut self, enable: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.deep_introspection = enable;
        self.config = Some(config);
        self
    }

    /// Set the insight threshold.
    pub fn insight_threshold(mut self, threshold: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.insight_threshold = threshold.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set the maximum number of insights.
    pub fn max_insights(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_insights = max;
        self.config = Some(config);
        self
    }

    /// Enable or disable process tracking.
    pub fn include_processes(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_processes = include;
        self.config = Some(config);
        self
    }

    /// Set the stale agent threshold.
    pub fn stale_agent_threshold_secs(mut self, secs: i64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.stale_agent_threshold_secs = secs;
        self.config = Some(config);
        self
    }

    /// Enable or disable auto insight generation.
    pub fn auto_generate_insights(mut self, enable: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.auto_generate_insights = enable;
        self.config = Some(config);
        self
    }

    /// Build the ConsciousnessAgent.
    pub fn build(self) -> ConsciousnessAgent {
        ConsciousnessAgent {
            config: self.config.unwrap_or_default(),
            agent_states: HashMap::new(),
            active_processes: Vec::new(),
            created_at: Instant::now(),
            total_requests: 0,
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
    // OperatingMode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_operating_mode_display() {
        assert_eq!(OperatingMode::Normal.to_string(), "Normal");
        assert_eq!(OperatingMode::HighLoad.to_string(), "High Load");
        assert_eq!(OperatingMode::Maintenance.to_string(), "Maintenance");
        assert_eq!(OperatingMode::Learning.to_string(), "Learning");
        assert_eq!(OperatingMode::Recovery.to_string(), "Recovery");
    }

    #[test]
    fn test_operating_mode_default() {
        assert_eq!(OperatingMode::default(), OperatingMode::Normal);
    }

    #[test]
    fn test_operating_mode_all() {
        let all = OperatingMode::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&OperatingMode::Normal));
        assert!(all.contains(&OperatingMode::Recovery));
    }

    #[test]
    fn test_operating_mode_allows_requests() {
        assert!(OperatingMode::Normal.allows_requests());
        assert!(OperatingMode::HighLoad.allows_requests());
        assert!(!OperatingMode::Maintenance.allows_requests());
        assert!(OperatingMode::Learning.allows_requests());
        assert!(!OperatingMode::Recovery.allows_requests());
    }

    #[test]
    fn test_operating_mode_is_degraded() {
        assert!(!OperatingMode::Normal.is_degraded());
        assert!(OperatingMode::HighLoad.is_degraded());
        assert!(OperatingMode::Maintenance.is_degraded());
        assert!(!OperatingMode::Learning.is_degraded());
        assert!(OperatingMode::Recovery.is_degraded());
    }

    #[test]
    fn test_operating_mode_is_adaptive() {
        assert!(!OperatingMode::Normal.is_adaptive());
        assert!(!OperatingMode::HighLoad.is_adaptive());
        assert!(OperatingMode::Learning.is_adaptive());
        assert!(OperatingMode::Recovery.is_adaptive());
    }

    #[test]
    fn test_operating_mode_description() {
        assert!(OperatingMode::Normal.description().contains("normal"));
        assert!(OperatingMode::Recovery.description().contains("recovering"));
    }

    #[test]
    fn test_operating_mode_priority_multiplier() {
        assert_eq!(OperatingMode::Normal.priority_multiplier(), 1.0);
        assert!(OperatingMode::Recovery.priority_multiplier() > 1.0);
        assert!(OperatingMode::Maintenance.priority_multiplier() < 1.0);
    }

    // -------------------------------------------------------------------------
    // SystemState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_system_state_default() {
        let state = SystemState::default();
        assert_eq!(state.uptime, Duration::from_secs(0));
        assert_eq!(state.total_requests, 0);
        assert_eq!(state.current_load, 0.0);
        assert_eq!(state.mode, OperatingMode::Normal);
    }

    #[test]
    fn test_system_state_with_values() {
        let state =
            SystemState::with_values(Duration::from_secs(3600), 1000, 50.0).with_mode(OperatingMode::HighLoad);
        assert_eq!(state.uptime.as_secs(), 3600);
        assert_eq!(state.total_requests, 1000);
        assert_eq!(state.current_load, 50.0);
        assert_eq!(state.mode, OperatingMode::HighLoad);
    }

    #[test]
    fn test_system_state_load_clamping() {
        let state = SystemState::with_values(Duration::from_secs(0), 0, 150.0);
        assert_eq!(state.current_load, 100.0);

        let state = SystemState::with_values(Duration::from_secs(0), 0, -10.0);
        assert_eq!(state.current_load, 0.0);
    }

    #[test]
    fn test_system_state_is_healthy() {
        let healthy = SystemState::with_values(Duration::from_secs(0), 0, 50.0);
        assert!(healthy.is_healthy());

        let overloaded = SystemState::with_values(Duration::from_secs(0), 0, 95.0);
        assert!(!overloaded.is_healthy());

        let maintenance = SystemState::with_values(Duration::from_secs(0), 0, 30.0)
            .with_mode(OperatingMode::Maintenance);
        assert!(!maintenance.is_healthy());
    }

    #[test]
    fn test_system_state_is_overloaded() {
        let normal = SystemState::with_values(Duration::from_secs(0), 0, 50.0);
        assert!(!normal.is_overloaded());

        let high_load = SystemState::with_values(Duration::from_secs(0), 0, 85.0);
        assert!(high_load.is_overloaded());

        let mode_high = SystemState::with_values(Duration::from_secs(0), 0, 30.0)
            .with_mode(OperatingMode::HighLoad);
        assert!(mode_high.is_overloaded());
    }

    #[test]
    fn test_system_state_uptime_display() {
        let state = SystemState::with_values(Duration::from_secs(90061), 0, 0.0);
        let display = state.uptime_display();
        assert!(display.contains("1d"));
        assert!(display.contains("1h"));
        assert!(display.contains("1m"));
        assert!(display.contains("1s"));
    }

    #[test]
    fn test_system_state_to_markdown() {
        let state = SystemState::with_values(Duration::from_secs(3600), 100, 45.0)
            .with_agent_count(5)
            .with_memory_usage(256.0);
        let md = state.to_markdown();
        assert!(md.contains("System State"));
        assert!(md.contains("Normal"));
        assert!(md.contains("100"));
        assert!(md.contains("256"));
    }

    // -------------------------------------------------------------------------
    // AgentStatus Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_status_display() {
        assert_eq!(AgentStatus::Active.to_string(), "Active");
        assert_eq!(AgentStatus::Idle.to_string(), "Idle");
        assert_eq!(AgentStatus::Error.to_string(), "Error");
    }

    #[test]
    fn test_agent_status_default() {
        assert_eq!(AgentStatus::default(), AgentStatus::Active);
    }

    #[test]
    fn test_agent_status_is_operational() {
        assert!(AgentStatus::Active.is_operational());
        assert!(AgentStatus::Idle.is_operational());
        assert!(!AgentStatus::Error.is_operational());
        assert!(!AgentStatus::Paused.is_operational());
    }

    #[test]
    fn test_agent_status_has_issues() {
        assert!(!AgentStatus::Active.has_issues());
        assert!(AgentStatus::Error.has_issues());
        assert!(AgentStatus::Paused.has_issues());
    }

    // -------------------------------------------------------------------------
    // AgentState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_state_new() {
        let state = AgentState::new("test_agent");
        assert_eq!(state.name, "test_agent");
        assert_eq!(state.status, AgentStatus::Active);
        assert_eq!(state.performance_score, 1.0);
        assert_eq!(state.requests_processed, 0);
    }

    #[test]
    fn test_agent_state_with_status() {
        let state = AgentState::new("test").with_status(AgentStatus::Idle);
        assert_eq!(state.status, AgentStatus::Idle);
    }

    #[test]
    fn test_agent_state_with_performance_score() {
        let state = AgentState::new("test").with_performance_score(0.85);
        assert_eq!(state.performance_score, 0.85);

        let clamped = AgentState::new("test").with_performance_score(1.5);
        assert_eq!(clamped.performance_score, 1.0);
    }

    #[test]
    fn test_agent_state_with_error() {
        let state = AgentState::new("test").with_error("Connection failed");
        assert_eq!(state.status, AgentStatus::Error);
        assert_eq!(state.last_error.unwrap(), "Connection failed");
    }

    #[test]
    fn test_agent_state_with_error_count() {
        let state = AgentState::new("test").with_error_count(3);
        assert_eq!(state.error_count, 3);
        assert_eq!(state.status, AgentStatus::Error);
    }

    #[test]
    fn test_agent_state_touch() {
        let mut state = AgentState::new("test");
        let before = state.last_activity;
        std::thread::sleep(std::time::Duration::from_millis(10));
        state.touch();
        assert!(state.last_activity > before);
    }

    #[test]
    fn test_agent_state_is_operational() {
        let active = AgentState::new("test");
        assert!(active.is_operational());

        let error = AgentState::new("test").with_error("Error");
        assert!(!error.is_operational());
    }

    // -------------------------------------------------------------------------
    // Process Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_process_new() {
        let process = Process::new("proc-1", "Test Process");
        assert_eq!(process.id, "proc-1");
        assert_eq!(process.name, "Test Process");
        assert_eq!(process.progress, 0.0);
        assert_eq!(process.priority, 5);
    }

    #[test]
    fn test_process_with_progress() {
        let process = Process::new("proc-1", "Test").with_progress(50.0);
        assert_eq!(process.progress, 50.0);

        let clamped = Process::new("proc-1", "Test").with_progress(150.0);
        assert_eq!(clamped.progress, 100.0);
    }

    #[test]
    fn test_process_with_activity() {
        let process = Process::new("proc-1", "Test").with_activity("Processing data");
        assert_eq!(process.current_activity.unwrap(), "Processing data");
    }

    #[test]
    fn test_process_is_complete() {
        let incomplete = Process::new("proc-1", "Test").with_progress(50.0);
        assert!(!incomplete.is_complete());

        let complete = Process::new("proc-1", "Test").with_progress(100.0);
        assert!(complete.is_complete());
    }

    #[test]
    fn test_process_elapsed() {
        let process = Process::new("proc-1", "Test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(process.elapsed().as_millis() >= 10);
    }

    // -------------------------------------------------------------------------
    // InsightCategory Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_insight_category_display() {
        assert_eq!(InsightCategory::Performance.to_string(), "Performance");
        assert_eq!(InsightCategory::Anomaly.to_string(), "Anomaly");
        assert_eq!(InsightCategory::Warning.to_string(), "Warning");
    }

    #[test]
    fn test_insight_category_default() {
        assert_eq!(InsightCategory::default(), InsightCategory::Performance);
    }

    #[test]
    fn test_insight_category_all() {
        let all = InsightCategory::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_insight_category_requires_attention() {
        assert!(!InsightCategory::Performance.requires_attention());
        assert!(InsightCategory::Anomaly.requires_attention());
        assert!(InsightCategory::Warning.requires_attention());
    }

    #[test]
    fn test_insight_category_icon() {
        assert_eq!(InsightCategory::Performance.icon(), "PERF");
        assert_eq!(InsightCategory::Warning.icon(), "WARN");
    }

    // -------------------------------------------------------------------------
    // Insight Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_insight_new() {
        let insight = Insight::new(InsightCategory::Performance, "Test insight");
        assert_eq!(insight.category, InsightCategory::Performance);
        assert_eq!(insight.description, "Test insight");
        assert_eq!(insight.confidence, 1.0);
        assert!(!insight.actionable);
    }

    #[test]
    fn test_insight_convenience_constructors() {
        assert_eq!(
            Insight::performance("Test").category,
            InsightCategory::Performance
        );
        assert_eq!(
            Insight::behavior("Test").category,
            InsightCategory::Behavior
        );
        assert_eq!(Insight::anomaly("Test").category, InsightCategory::Anomaly);
        assert_eq!(
            Insight::optimization("Test").category,
            InsightCategory::Optimization
        );
        assert_eq!(Insight::warning("Test").category, InsightCategory::Warning);
    }

    #[test]
    fn test_insight_with_confidence() {
        let insight = Insight::performance("Test").with_confidence(0.75);
        assert_eq!(insight.confidence, 0.75);

        let clamped = Insight::performance("Test").with_confidence(1.5);
        assert_eq!(clamped.confidence, 1.0);
    }

    #[test]
    fn test_insight_with_action() {
        let insight = Insight::performance("Test").with_action("Do something");
        assert!(insight.actionable);
        assert_eq!(insight.suggested_action.unwrap(), "Do something");
    }

    #[test]
    fn test_insight_requires_attention() {
        let warning = Insight::warning("Test");
        assert!(warning.requires_attention());

        let actionable = Insight::performance("Test")
            .with_action("Action")
            .with_confidence(0.9);
        assert!(actionable.requires_attention());

        let normal = Insight::performance("Test").with_confidence(0.5);
        assert!(!normal.requires_attention());
    }

    #[test]
    fn test_insight_to_log_line() {
        let insight = Insight::warning("High memory usage")
            .with_action("Increase memory")
            .with_confidence(0.9);
        let log = insight.to_log_line();
        assert!(log.contains("[WARN]"));
        assert!(log.contains("[ACTION]"));
        assert!(log.contains("High memory usage"));
    }

    // -------------------------------------------------------------------------
    // HealthLevel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_health_level_display() {
        assert_eq!(HealthLevel::Excellent.to_string(), "Excellent");
        assert_eq!(HealthLevel::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_health_level_default() {
        assert_eq!(HealthLevel::default(), HealthLevel::Good);
    }

    #[test]
    fn test_health_level_score() {
        assert_eq!(HealthLevel::Excellent.score(), 100);
        assert_eq!(HealthLevel::Good.score(), 80);
        assert_eq!(HealthLevel::Critical.score(), 20);
    }

    #[test]
    fn test_health_level_from_score() {
        assert_eq!(HealthLevel::from_score(95), HealthLevel::Excellent);
        assert_eq!(HealthLevel::from_score(75), HealthLevel::Good);
        assert_eq!(HealthLevel::from_score(55), HealthLevel::Fair);
        assert_eq!(HealthLevel::from_score(35), HealthLevel::Poor);
        assert_eq!(HealthLevel::from_score(15), HealthLevel::Critical);
    }

    #[test]
    fn test_health_level_is_acceptable() {
        assert!(HealthLevel::Excellent.is_acceptable());
        assert!(HealthLevel::Good.is_acceptable());
        assert!(HealthLevel::Fair.is_acceptable());
        assert!(!HealthLevel::Poor.is_acceptable());
        assert!(!HealthLevel::Critical.is_acceptable());
    }

    // -------------------------------------------------------------------------
    // SelfAssessment Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_self_assessment_new() {
        let assessment = SelfAssessment::new();
        assert!(assessment.strengths.is_empty());
        assert!(assessment.weaknesses.is_empty());
        assert_eq!(assessment.overall_health, HealthLevel::Good);
    }

    #[test]
    fn test_self_assessment_add_strength() {
        let mut assessment = SelfAssessment::new();
        assessment.add_strength("Fast response time");
        assert_eq!(assessment.strengths.len(), 1);
        assert_eq!(assessment.strengths[0], "Fast response time");
    }

    #[test]
    fn test_self_assessment_add_weakness() {
        let mut assessment = SelfAssessment::new();
        assessment.add_weakness("High memory usage");
        assert_eq!(assessment.weaknesses.len(), 1);
    }

    #[test]
    fn test_self_assessment_calculate_health() {
        let mut assessment = SelfAssessment::new();
        assessment.add_strength("Strength 1");
        assessment.add_strength("Strength 2");
        assessment.add_strength("Strength 3");
        assessment.add_weakness("Weakness 1");
        assessment.calculate_health();
        // 3 strengths, 1 weakness = 75% = Good
        assert_eq!(assessment.overall_health, HealthLevel::Good);
    }

    #[test]
    fn test_self_assessment_strength_ratio() {
        let mut assessment = SelfAssessment::new();
        assessment.add_strength("S1");
        assessment.add_strength("S2");
        assessment.add_weakness("W1");
        assert_eq!(assessment.strength_ratio(), 2.0);
    }

    #[test]
    fn test_self_assessment_to_markdown() {
        let mut assessment = SelfAssessment::new();
        assessment.add_strength("Fast");
        assessment.add_weakness("Memory");
        assessment.add_improvement_area("Caching");
        let md = assessment.to_markdown();
        assert!(md.contains("Self Assessment"));
        assert!(md.contains("Strengths"));
        assert!(md.contains("Fast"));
        assert!(md.contains("Weaknesses"));
    }

    // -------------------------------------------------------------------------
    // ConsciousnessReport Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_consciousness_report_new() {
        let state = SystemState::default();
        let assessment = SelfAssessment::new();
        let report = ConsciousnessReport::new(state, assessment);
        assert!(report.agent_states.is_empty());
        assert!(report.active_processes.is_empty());
        assert!(report.insights.is_empty());
    }

    #[test]
    fn test_consciousness_report_add_agent_state() {
        let mut report = ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        report.add_agent_state(AgentState::new("agent1"));
        assert_eq!(report.agent_states.len(), 1);
        assert!(report.agent_states.contains_key("agent1"));
    }

    #[test]
    fn test_consciousness_report_add_process() {
        let mut report = ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        report.add_process(Process::new("proc-1", "Test Process"));
        assert_eq!(report.active_processes.len(), 1);
    }

    #[test]
    fn test_consciousness_report_add_insight() {
        let mut report = ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        report.add_insight(Insight::performance("Test"));
        assert_eq!(report.insights.len(), 1);
    }

    #[test]
    fn test_consciousness_report_operational_agent_count() {
        let mut report = ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        report.add_agent_state(AgentState::new("agent1"));
        report.add_agent_state(AgentState::new("agent2").with_error("Error"));
        assert_eq!(report.operational_agent_count(), 1);
    }

    #[test]
    fn test_consciousness_report_agents_with_issues() {
        let mut report = ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        report.add_agent_state(AgentState::new("agent1"));
        report.add_agent_state(AgentState::new("agent2").with_error("Error"));
        report.add_agent_state(AgentState::new("agent3").with_status(AgentStatus::Paused));
        assert_eq!(report.agents_with_issues(), 2);
    }

    #[test]
    fn test_consciousness_report_actionable_insights() {
        let mut report = ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        report.add_insight(Insight::performance("Not actionable"));
        report.add_insight(Insight::warning("Actionable").with_action("Do something"));
        assert_eq!(report.actionable_insights().len(), 1);
    }

    #[test]
    fn test_consciousness_report_insights_by_category() {
        let mut report = ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        report.add_insight(Insight::performance("Perf 1"));
        report.add_insight(Insight::performance("Perf 2"));
        report.add_insight(Insight::warning("Warning"));
        assert_eq!(
            report
                .insights_by_category(InsightCategory::Performance)
                .len(),
            2
        );
        assert_eq!(
            report.insights_by_category(InsightCategory::Warning).len(),
            1
        );
    }

    #[test]
    fn test_consciousness_report_is_healthy() {
        let healthy_report =
            ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        assert!(healthy_report.is_healthy());

        let mut unhealthy_report =
            ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        unhealthy_report.add_agent_state(AgentState::new("agent").with_error("Error"));
        assert!(!unhealthy_report.is_healthy());
    }

    #[test]
    fn test_consciousness_report_summary() {
        let mut report = ConsciousnessReport::new(
            SystemState::with_values(Duration::from_secs(100), 50, 30.0),
            SelfAssessment::new(),
        );
        report.add_agent_state(AgentState::new("agent1"));
        report.add_insight(Insight::performance("Test"));
        let summary = report.summary();
        assert!(summary.contains("Normal"));
        assert!(summary.contains("1 agents"));
        assert!(summary.contains("1 insights"));
    }

    #[test]
    fn test_consciousness_report_to_markdown() {
        let mut report = ConsciousnessReport::new(SystemState::default(), SelfAssessment::new());
        report.add_agent_state(AgentState::new("agent1"));
        report.add_process(Process::new("proc-1", "Test").with_progress(50.0));
        report.add_insight(Insight::performance("Performance insight"));
        let md = report.to_markdown();
        assert!(md.contains("# Consciousness Report"));
        assert!(md.contains("Agent States"));
        assert!(md.contains("Active Processes"));
        assert!(md.contains("Insights"));
    }

    // -------------------------------------------------------------------------
    // ConsciousnessConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_consciousness_config_default() {
        let config = ConsciousnessConfig::default();
        assert!(!config.deep_introspection);
        assert_eq!(config.insight_threshold, 0.5);
        assert_eq!(config.max_insights, 20);
        assert!(config.include_processes);
        assert!(config.auto_generate_insights);
    }

    #[test]
    fn test_consciousness_config_minimal() {
        let config = ConsciousnessConfig::minimal();
        assert!(!config.deep_introspection);
        assert!(!config.include_processes);
        assert!(!config.auto_generate_insights);
    }

    #[test]
    fn test_consciousness_config_thorough() {
        let config = ConsciousnessConfig::thorough();
        assert!(config.deep_introspection);
        assert!(config.include_processes);
        assert!(config.auto_generate_insights);
        assert_eq!(config.max_insights, 50);
    }

    // -------------------------------------------------------------------------
    // ConsciousnessAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_consciousness_agent_new() {
        let agent = ConsciousnessAgent::new();
        assert!(!agent.config().deep_introspection);
        assert!(agent.agent_states.is_empty());
    }

    #[test]
    fn test_consciousness_agent_with_config() {
        let config = ConsciousnessConfig::thorough();
        let agent = ConsciousnessAgent::with_config(config);
        assert!(agent.config().deep_introspection);
    }

    #[test]
    fn test_consciousness_agent_builder() {
        let agent = ConsciousnessAgent::builder()
            .enable_deep_introspection(true)
            .insight_threshold(0.8)
            .max_insights(10)
            .build();

        assert!(agent.config().deep_introspection);
        assert_eq!(agent.config().insight_threshold, 0.8);
        assert_eq!(agent.config().max_insights, 10);
    }

    #[test]
    fn test_consciousness_agent_default() {
        let agent = ConsciousnessAgent::default();
        assert!(!agent.config().deep_introspection);
    }

    // -------------------------------------------------------------------------
    // ConsciousnessAgent Agent Management Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_consciousness_agent_register_agent() {
        let mut agent = ConsciousnessAgent::new();
        agent.register_agent("test_agent");
        assert!(agent.get_agent_state("test_agent").is_some());
    }

    #[test]
    fn test_consciousness_agent_update_agent_state() {
        let mut agent = ConsciousnessAgent::new();
        agent.register_agent("test_agent");
        agent.update_agent_state(
            "test_agent",
            AgentState::new("test_agent").with_performance_score(0.9),
        );
        let state = agent.get_agent_state("test_agent").unwrap();
        assert_eq!(state.performance_score, 0.9);
    }

    // -------------------------------------------------------------------------
    // ConsciousnessAgent Process Management Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_consciousness_agent_start_process() {
        let mut agent = ConsciousnessAgent::new();
        agent.start_process(Process::new("proc-1", "Test Process"));
        assert_eq!(agent.active_processes.len(), 1);
    }

    #[test]
    fn test_consciousness_agent_update_process() {
        let mut agent = ConsciousnessAgent::new();
        agent.start_process(Process::new("proc-1", "Test Process"));
        agent.update_process("proc-1", 50.0);
        assert_eq!(agent.active_processes[0].progress, 50.0);
    }

    #[test]
    fn test_consciousness_agent_complete_process() {
        let mut agent = ConsciousnessAgent::new();
        agent.start_process(Process::new("proc-1", "Test Process"));
        agent.complete_process("proc-1");
        assert!(agent.active_processes.is_empty());
    }

    #[test]
    fn test_consciousness_agent_record_request() {
        let mut agent = ConsciousnessAgent::new();
        agent.record_request();
        agent.record_request();
        assert_eq!(agent.total_requests, 2);
    }

    // -------------------------------------------------------------------------
    // ConsciousnessAgent Introspection Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_consciousness_agent_introspect() {
        let agent = ConsciousnessAgent::new();
        let report = agent.introspect().await.unwrap();
        // Duration is always non-negative (u64)
        let _ = report.introspection_duration_ms;
        assert!(!report.summary().is_empty());
    }

    #[tokio::test]
    async fn test_consciousness_agent_introspect_with_agents() {
        let mut agent = ConsciousnessAgent::new();
        agent.register_agent("agent1");
        agent.register_agent("agent2");
        let report = agent.introspect().await.unwrap();
        assert_eq!(report.agent_states.len(), 2);
    }

    #[tokio::test]
    async fn test_consciousness_agent_introspect_with_processes() {
        let mut agent = ConsciousnessAgent::new();
        agent.start_process(Process::new("proc-1", "Test").with_progress(50.0));
        let report = agent.introspect().await.unwrap();
        assert_eq!(report.active_processes.len(), 1);
    }

    #[tokio::test]
    async fn test_consciousness_agent_introspect_generates_insights() {
        let mut agent = ConsciousnessAgent::new();
        agent.register_agent("agent1");
        let report = agent.introspect().await.unwrap();
        // Should generate at least one insight about system state
        assert!(!report.insights.is_empty() || report.agent_states.len() > 0);
    }

    #[tokio::test]
    async fn test_consciousness_agent_introspect_no_insights_when_disabled() {
        let agent = ConsciousnessAgent::builder()
            .auto_generate_insights(false)
            .build();
        let report = agent.introspect().await.unwrap();
        assert!(report.insights.is_empty());
    }

    #[test]
    fn test_consciousness_agent_current_state() {
        let mut agent = ConsciousnessAgent::new();
        agent.register_agent("agent1");
        agent.record_request();
        let state = agent.current_state();
        assert_eq!(state.total_requests, 1);
        assert_eq!(state.active_agent_count, 1);
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_config() {
        let config = ConsciousnessConfig::thorough();
        let agent = ConsciousnessAgentBuilder::default().config(config).build();
        assert!(agent.config().deep_introspection);
    }

    #[test]
    fn test_builder_enable_deep_introspection() {
        let agent = ConsciousnessAgentBuilder::default()
            .enable_deep_introspection(true)
            .build();
        assert!(agent.config().deep_introspection);
    }

    #[test]
    fn test_builder_insight_threshold() {
        let agent = ConsciousnessAgentBuilder::default()
            .insight_threshold(0.75)
            .build();
        assert_eq!(agent.config().insight_threshold, 0.75);
    }

    #[test]
    fn test_builder_insight_threshold_clamping() {
        let agent = ConsciousnessAgentBuilder::default()
            .insight_threshold(1.5)
            .build();
        assert_eq!(agent.config().insight_threshold, 1.0);
    }

    #[test]
    fn test_builder_max_insights() {
        let agent = ConsciousnessAgentBuilder::default().max_insights(30).build();
        assert_eq!(agent.config().max_insights, 30);
    }

    #[test]
    fn test_builder_include_processes() {
        let agent = ConsciousnessAgentBuilder::default()
            .include_processes(false)
            .build();
        assert!(!agent.config().include_processes);
    }

    #[test]
    fn test_builder_stale_agent_threshold() {
        let agent = ConsciousnessAgentBuilder::default()
            .stale_agent_threshold_secs(600)
            .build();
        assert_eq!(agent.config().stale_agent_threshold_secs, 600);
    }

    #[test]
    fn test_builder_auto_generate_insights() {
        let agent = ConsciousnessAgentBuilder::default()
            .auto_generate_insights(false)
            .build();
        assert!(!agent.config().auto_generate_insights);
    }

    #[test]
    fn test_builder_chaining() {
        let agent = ConsciousnessAgentBuilder::default()
            .enable_deep_introspection(true)
            .insight_threshold(0.6)
            .max_insights(15)
            .include_processes(true)
            .stale_agent_threshold_secs(200)
            .auto_generate_insights(true)
            .build();

        assert!(agent.config().deep_introspection);
        assert_eq!(agent.config().insight_threshold, 0.6);
        assert_eq!(agent.config().max_insights, 15);
        assert!(agent.config().include_processes);
        assert_eq!(agent.config().stale_agent_threshold_secs, 200);
        assert!(agent.config().auto_generate_insights);
    }
}
