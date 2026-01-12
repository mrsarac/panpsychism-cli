//! Transcender Agent module for Project Panpsychism.
//!
//! Agent 40: The Ultimate Orchestrator - "From chaos, forge cosmic order."
//!
//! This module implements the Transcender Agent, the supreme orchestrator that
//! coordinates all tiers of agents with ultimate coordination, tier management,
//! and system evolution. Like a cosmic conductor, the Transcender harmonizes
//! the entire agent ecosystem into a unified, self-evolving system.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, the Transcender Agent serves as the Grand Archmage:
//!
//! - **Goals** (cosmic intentions) define the ultimate objectives
//! - **Orchestration Plans** (the grand spell design) coordinate all agents
//! - **Phases** (ritual stages) execute in optimal order
//! - **Transcendence** (apotheosis) is the synthesis of all agent wisdom
//!
//! The Transcender orchestrates:
//! - Multi-tier agent coordination with parallel execution
//! - Dynamic orchestration plan generation based on goals
//! - Fallback path management for resilient execution
//! - Quality-speed-cost optimization balancing
//! - System-wide evolution and learning
//!
//! ## Philosophy
//!
//! Following Spinoza's principles at the highest level:
//!
//! - **CONATUS**: The drive to optimize and evolve the entire system
//! - **NATURA**: Natural harmony between all agents and tiers
//! - **RATIO**: Supreme logical orchestration across all components
//! - **LAETITIA**: Joy through transcendent synthesis and emergence
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::transcender::{
//!     TranscenderAgent, TranscendentGoal, OptimizationTarget, Constraint
//! };
//!
//! let transcender = TranscenderAgent::builder()
//!     .parallel_execution(true)
//!     .timeout_ms(30000)
//!     .build();
//!
//! let goal = TranscendentGoal::new("Generate a comprehensive analysis")
//!     .with_constraint(Constraint::new("max_latency_ms", "number", 5000))
//!     .with_optimization_target(OptimizationTarget::Balanced)
//!     .with_time_budget_ms(10000);
//!
//! let response = transcender.transcend(&goal).await?;
//! println!("Result: {}", response.response);
//! println!("Quality: {:.2}", response.quality_score);
//! println!("Agents used: {:?}", response.agents_used);
//! ```

use crate::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info, warn};

// =============================================================================
// OPTIMIZATION TARGET ENUM
// =============================================================================

/// The optimization target for transcendent orchestration.
///
/// Defines what the Transcender should prioritize when orchestrating agents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Prioritize response quality above all else.
    ///
    /// Uses more agents, longer processing, and extensive validation.
    Quality,

    /// Prioritize speed of response.
    ///
    /// Uses minimal agents and parallel execution for fastest results.
    Speed,

    /// Prioritize cost efficiency.
    ///
    /// Uses fewest agents and simplest orchestration paths.
    Cost,

    /// Balance quality, speed, and cost equally.
    ///
    /// Default mode for general-purpose orchestration.
    Balanced,

    /// Custom weighted optimization.
    ///
    /// Allows fine-grained control over optimization priorities.
    /// Vec contains (dimension_name, weight) pairs where weights sum to 1.0.
    Custom(Vec<(String, f64)>),
}

impl Default for OptimizationTarget {
    fn default() -> Self {
        Self::Balanced
    }
}

impl std::fmt::Display for OptimizationTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Quality => write!(f, "quality"),
            Self::Speed => write!(f, "speed"),
            Self::Cost => write!(f, "cost"),
            Self::Balanced => write!(f, "balanced"),
            Self::Custom(weights) => {
                let dims: Vec<String> = weights
                    .iter()
                    .map(|(name, weight)| format!("{}:{:.2}", name, weight))
                    .collect();
                write!(f, "custom({})", dims.join(","))
            }
        }
    }
}

impl std::str::FromStr for OptimizationTarget {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "quality" | "high_quality" | "best" => Ok(Self::Quality),
            "speed" | "fast" | "quick" => Ok(Self::Speed),
            "cost" | "cheap" | "efficient" => Ok(Self::Cost),
            "balanced" | "default" | "normal" => Ok(Self::Balanced),
            s if s.starts_with("custom(") => {
                // Parse custom weights: custom(quality:0.5,speed:0.3,cost:0.2)
                let inner = s
                    .strip_prefix("custom(")
                    .and_then(|s| s.strip_suffix(')'))
                    .ok_or_else(|| Error::Config("Invalid custom optimization format".into()))?;

                let weights: Vec<(String, f64)> = inner
                    .split(',')
                    .filter_map(|pair| {
                        let parts: Vec<&str> = pair.split(':').collect();
                        if parts.len() == 2 {
                            parts[1]
                                .parse::<f64>()
                                .ok()
                                .map(|w| (parts[0].to_string(), w))
                        } else {
                            None
                        }
                    })
                    .collect();

                if weights.is_empty() {
                    return Err(Error::Config("No valid weights in custom optimization".into()));
                }

                Ok(Self::Custom(weights))
            }
            _ => Err(Error::Config(format!(
                "Unknown optimization target: '{}'. Valid: quality, speed, cost, balanced, custom(...)",
                s
            ))),
        }
    }
}

impl OptimizationTarget {
    /// Get the quality weight for this optimization target.
    pub fn quality_weight(&self) -> f64 {
        match self {
            Self::Quality => 1.0,
            Self::Speed => 0.2,
            Self::Cost => 0.3,
            Self::Balanced => 0.5,
            Self::Custom(weights) => weights
                .iter()
                .find(|(n, _)| n == "quality")
                .map(|(_, w)| *w)
                .unwrap_or(0.33),
        }
    }

    /// Get the speed weight for this optimization target.
    pub fn speed_weight(&self) -> f64 {
        match self {
            Self::Quality => 0.2,
            Self::Speed => 1.0,
            Self::Cost => 0.3,
            Self::Balanced => 0.5,
            Self::Custom(weights) => weights
                .iter()
                .find(|(n, _)| n == "speed")
                .map(|(_, w)| *w)
                .unwrap_or(0.33),
        }
    }

    /// Get the cost weight for this optimization target.
    pub fn cost_weight(&self) -> f64 {
        match self {
            Self::Quality => 0.2,
            Self::Speed => 0.2,
            Self::Cost => 1.0,
            Self::Balanced => 0.5,
            Self::Custom(weights) => weights
                .iter()
                .find(|(n, _)| n == "cost")
                .map(|(_, w)| *w)
                .unwrap_or(0.33),
        }
    }

    /// Get all available optimization targets.
    pub fn all_standard() -> Vec<Self> {
        vec![Self::Quality, Self::Speed, Self::Cost, Self::Balanced]
    }

    /// Create a custom optimization target with specified weights.
    pub fn custom(quality: f64, speed: f64, cost: f64) -> Self {
        Self::Custom(vec![
            ("quality".to_string(), quality),
            ("speed".to_string(), speed),
            ("cost".to_string(), cost),
        ])
    }
}

// =============================================================================
// CONSTRAINT STRUCT
// =============================================================================

/// A constraint that limits or guides the transcendent orchestration.
///
/// Constraints define boundaries and requirements that the Transcender
/// must respect when orchestrating agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// The name of the constraint (e.g., "max_latency_ms", "required_agents").
    pub name: String,

    /// The type of the constraint value (e.g., "number", "string", "boolean").
    pub constraint_type: String,

    /// The constraint value as a string (parsed based on constraint_type).
    pub value: String,

    /// Optional description of the constraint.
    pub description: Option<String>,

    /// Whether this constraint is strict (must be satisfied) or soft (preferred).
    pub strict: bool,
}

impl Constraint {
    /// Create a new constraint.
    pub fn new(
        name: impl Into<String>,
        constraint_type: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            constraint_type: constraint_type.into(),
            value: value.into(),
            description: None,
            strict: true,
        }
    }

    /// Set the description for this constraint.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set whether this constraint is strict.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Parse the value as a number.
    pub fn as_number(&self) -> Option<f64> {
        self.value.parse().ok()
    }

    /// Parse the value as a boolean.
    pub fn as_boolean(&self) -> Option<bool> {
        match self.value.to_lowercase().as_str() {
            "true" | "yes" | "1" => Some(true),
            "false" | "no" | "0" => Some(false),
            _ => None,
        }
    }

    /// Parse the value as a list of strings.
    pub fn as_list(&self) -> Vec<String> {
        self.value
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Common constraint: maximum latency in milliseconds.
    pub fn max_latency_ms(ms: u64) -> Self {
        Self::new("max_latency_ms", "number", ms.to_string())
            .with_description("Maximum allowed latency in milliseconds")
    }

    /// Common constraint: minimum quality score.
    pub fn min_quality(score: f64) -> Self {
        Self::new("min_quality", "number", score.to_string())
            .with_description("Minimum acceptable quality score (0.0-1.0)")
    }

    /// Common constraint: required agents.
    pub fn required_agents(agents: &[&str]) -> Self {
        Self::new("required_agents", "list", agents.join(","))
            .with_description("Agents that must be used in orchestration")
    }

    /// Common constraint: excluded agents.
    pub fn excluded_agents(agents: &[&str]) -> Self {
        Self::new("excluded_agents", "list", agents.join(","))
            .with_description("Agents that must not be used in orchestration")
    }

    /// Common constraint: maximum cost.
    pub fn max_cost(cost: f64) -> Self {
        Self::new("max_cost", "number", cost.to_string())
            .with_description("Maximum allowed cost for orchestration")
    }
}

impl Default for Constraint {
    fn default() -> Self {
        Self::new("", "", "")
    }
}

// =============================================================================
// TRANSCENDENT GOAL
// =============================================================================

/// A goal for transcendent orchestration.
///
/// Defines the objective, constraints, and optimization parameters
/// for a transcendent operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscendentGoal {
    /// The objective to achieve (e.g., "Generate a comprehensive analysis").
    pub objective: String,

    /// Constraints that limit or guide the orchestration.
    pub constraints: Vec<Constraint>,

    /// The optimization target for this goal.
    pub optimization_target: OptimizationTarget,

    /// Time budget in milliseconds for the entire operation.
    pub time_budget_ms: u64,

    /// Additional context for the goal.
    pub context: HashMap<String, String>,

    /// Priority level (1-10, higher is more important).
    pub priority: u8,

    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl TranscendentGoal {
    /// Create a new transcendent goal.
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            objective: objective.into(),
            constraints: Vec::new(),
            optimization_target: OptimizationTarget::default(),
            time_budget_ms: 30000,
            context: HashMap::new(),
            priority: 5,
            tags: Vec::new(),
        }
    }

    /// Add a constraint to the goal.
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add multiple constraints to the goal.
    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints.extend(constraints);
        self
    }

    /// Set the optimization target.
    pub fn with_optimization_target(mut self, target: OptimizationTarget) -> Self {
        self.optimization_target = target;
        self
    }

    /// Set the time budget in milliseconds.
    pub fn with_time_budget_ms(mut self, ms: u64) -> Self {
        self.time_budget_ms = ms;
        self
    }

    /// Add context to the goal.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Set the priority level.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }

    /// Add a tag to the goal.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags to the goal.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for tag in tags {
            self.tags.push(tag.into());
        }
        self
    }

    /// Get strict constraints.
    pub fn strict_constraints(&self) -> Vec<&Constraint> {
        self.constraints.iter().filter(|c| c.strict).collect()
    }

    /// Get soft constraints.
    pub fn soft_constraints(&self) -> Vec<&Constraint> {
        self.constraints.iter().filter(|c| !c.strict).collect()
    }

    /// Check if the goal has a specific constraint by name.
    pub fn has_constraint(&self, name: &str) -> bool {
        self.constraints.iter().any(|c| c.name == name)
    }

    /// Get a constraint by name.
    pub fn get_constraint(&self, name: &str) -> Option<&Constraint> {
        self.constraints.iter().find(|c| c.name == name)
    }
}

impl Default for TranscendentGoal {
    fn default() -> Self {
        Self::new("")
    }
}

// =============================================================================
// PHASE STRUCT
// =============================================================================

/// A phase in the orchestration plan.
///
/// Represents a stage of execution with specific agents and timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase {
    /// The name of this phase.
    pub name: String,

    /// Agents to execute in this phase.
    pub agents: Vec<String>,

    /// Timeout in milliseconds for this phase.
    pub timeout_ms: u64,

    /// Whether this phase is required (failure stops execution).
    pub required: bool,

    /// Order in which this phase should execute (lower = earlier).
    pub order: u32,

    /// Description of what this phase accomplishes.
    pub description: Option<String>,

    /// Dependencies on other phases (by name).
    pub depends_on: Vec<String>,
}

impl Phase {
    /// Create a new phase.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            agents: Vec::new(),
            timeout_ms: 5000,
            required: true,
            order: 0,
            description: None,
            depends_on: Vec::new(),
        }
    }

    /// Add an agent to this phase.
    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agents.push(agent.into());
        self
    }

    /// Add multiple agents to this phase.
    pub fn with_agents(mut self, agents: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for agent in agents {
            self.agents.push(agent.into());
        }
        self
    }

    /// Set the timeout for this phase.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Set whether this phase is required.
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set the execution order.
    pub fn with_order(mut self, order: u32) -> Self {
        self.order = order;
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a dependency on another phase.
    pub fn with_dependency(mut self, phase_name: impl Into<String>) -> Self {
        self.depends_on.push(phase_name.into());
        self
    }

    /// Check if this phase has dependencies.
    pub fn has_dependencies(&self) -> bool {
        !self.depends_on.is_empty()
    }

    /// Get the number of agents in this phase.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for Phase {
    fn default() -> Self {
        Self::new("")
    }
}

// =============================================================================
// FALLBACK PATH STRUCT
// =============================================================================

/// A fallback path for resilient execution.
///
/// Defines alternative agents to use when primary agents fail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackPath {
    /// Condition that triggers this fallback (e.g., "agent_timeout", "quality_below_threshold").
    pub condition: String,

    /// Alternative agents to use when the condition is met.
    pub alternative_agents: Vec<String>,

    /// Priority of this fallback (higher = preferred).
    pub priority: u8,

    /// Maximum retries before giving up.
    pub max_retries: u8,

    /// Description of when this fallback should be used.
    pub description: Option<String>,
}

impl FallbackPath {
    /// Create a new fallback path.
    pub fn new(condition: impl Into<String>) -> Self {
        Self {
            condition: condition.into(),
            alternative_agents: Vec::new(),
            priority: 5,
            max_retries: 3,
            description: None,
        }
    }

    /// Add an alternative agent.
    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.alternative_agents.push(agent.into());
        self
    }

    /// Add multiple alternative agents.
    pub fn with_agents(mut self, agents: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for agent in agents {
            self.alternative_agents.push(agent.into());
        }
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set the maximum retries.
    pub fn with_max_retries(mut self, retries: u8) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Common fallback: agent timeout.
    pub fn on_timeout(alternative_agents: &[&str]) -> Self {
        Self::new("agent_timeout")
            .with_agents(alternative_agents.iter().map(|s| s.to_string()))
            .with_description("Use faster alternative agents when timeout occurs")
    }

    /// Common fallback: quality below threshold.
    pub fn on_low_quality(alternative_agents: &[&str]) -> Self {
        Self::new("quality_below_threshold")
            .with_agents(alternative_agents.iter().map(|s| s.to_string()))
            .with_description("Use higher-quality agents when quality is insufficient")
    }

    /// Common fallback: agent error.
    pub fn on_error(alternative_agents: &[&str]) -> Self {
        Self::new("agent_error")
            .with_agents(alternative_agents.iter().map(|s| s.to_string()))
            .with_description("Use fallback agents when primary agents fail")
    }
}

impl Default for FallbackPath {
    fn default() -> Self {
        Self::new("")
    }
}

// =============================================================================
// ORCHESTRATION PLAN
// =============================================================================

/// The orchestration plan generated by the Transcender.
///
/// Defines how agents will be coordinated to achieve a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationPlan {
    /// Unique identifier for this plan.
    pub id: String,

    /// The phases to execute.
    pub phases: Vec<Phase>,

    /// Groups of agents that can execute in parallel.
    pub parallel_groups: Vec<Vec<String>>,

    /// Fallback paths for resilient execution.
    pub fallback_paths: Vec<FallbackPath>,

    /// Estimated total duration in milliseconds.
    pub estimated_duration_ms: u64,

    /// Estimated quality score (0.0-1.0).
    pub estimated_quality: f64,

    /// Estimated cost (arbitrary units).
    pub estimated_cost: f64,

    /// When this plan was created.
    pub created_at: DateTime<Utc>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl OrchestrationPlan {
    /// Create a new orchestration plan.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            phases: Vec::new(),
            parallel_groups: Vec::new(),
            fallback_paths: Vec::new(),
            estimated_duration_ms: 0,
            estimated_quality: 0.0,
            estimated_cost: 0.0,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add a phase to the plan.
    pub fn with_phase(mut self, phase: Phase) -> Self {
        self.phases.push(phase);
        self
    }

    /// Add multiple phases to the plan.
    pub fn with_phases(mut self, phases: Vec<Phase>) -> Self {
        self.phases.extend(phases);
        self
    }

    /// Add a parallel group.
    pub fn with_parallel_group(mut self, agents: Vec<String>) -> Self {
        self.parallel_groups.push(agents);
        self
    }

    /// Add a fallback path.
    pub fn with_fallback(mut self, fallback: FallbackPath) -> Self {
        self.fallback_paths.push(fallback);
        self
    }

    /// Set the estimated duration.
    pub fn with_estimated_duration_ms(mut self, ms: u64) -> Self {
        self.estimated_duration_ms = ms;
        self
    }

    /// Set the estimated quality.
    pub fn with_estimated_quality(mut self, quality: f64) -> Self {
        self.estimated_quality = quality.clamp(0.0, 1.0);
        self
    }

    /// Set the estimated cost.
    pub fn with_estimated_cost(mut self, cost: f64) -> Self {
        self.estimated_cost = cost;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get total agent count across all phases.
    pub fn total_agent_count(&self) -> usize {
        self.phases.iter().map(|p| p.agents.len()).sum()
    }

    /// Get unique agents used in this plan.
    pub fn unique_agents(&self) -> Vec<String> {
        let mut agents: Vec<String> = self
            .phases
            .iter()
            .flat_map(|p| p.agents.clone())
            .collect();
        agents.sort();
        agents.dedup();
        agents
    }

    /// Get required phases.
    pub fn required_phases(&self) -> Vec<&Phase> {
        self.phases.iter().filter(|p| p.required).collect()
    }

    /// Get optional phases.
    pub fn optional_phases(&self) -> Vec<&Phase> {
        self.phases.iter().filter(|p| !p.required).collect()
    }

    /// Get phases sorted by execution order.
    pub fn sorted_phases(&self) -> Vec<&Phase> {
        let mut phases: Vec<&Phase> = self.phases.iter().collect();
        phases.sort_by_key(|p| p.order);
        phases
    }

    /// Check if the plan has parallel groups.
    pub fn has_parallel_execution(&self) -> bool {
        !self.parallel_groups.is_empty()
    }

    /// Get a summary of the plan.
    pub fn summary(&self) -> String {
        format!(
            "Plan {}: {} phases, {} agents, ~{}ms, quality {:.2}, cost {:.2}",
            self.id,
            self.phases.len(),
            self.total_agent_count(),
            self.estimated_duration_ms,
            self.estimated_quality,
            self.estimated_cost
        )
    }
}

impl Default for OrchestrationPlan {
    fn default() -> Self {
        Self::new("")
    }
}

// =============================================================================
// TRANSCENDENT RESPONSE
// =============================================================================

/// The response from a transcendent operation.
///
/// Contains the result along with orchestration metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscendentResponse {
    /// The final synthesized response.
    pub response: String,

    /// The orchestration plan that was executed.
    pub orchestration_plan: OrchestrationPlan,

    /// Agents that were actually used.
    pub agents_used: Vec<String>,

    /// Total latency in milliseconds.
    pub total_latency_ms: u64,

    /// Quality score of the response (0.0-1.0).
    pub quality_score: f64,

    /// Phase execution results.
    pub phase_results: Vec<PhaseResult>,

    /// Any warnings generated during orchestration.
    pub warnings: Vec<String>,

    /// Any fallbacks that were triggered.
    pub fallbacks_triggered: Vec<String>,

    /// When the transcendence started.
    pub started_at: DateTime<Utc>,

    /// When the transcendence completed.
    pub completed_at: DateTime<Utc>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Result of executing a single phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    /// Phase name.
    pub phase_name: String,

    /// Whether the phase succeeded.
    pub success: bool,

    /// Duration in milliseconds.
    pub duration_ms: u64,

    /// Agents that executed in this phase.
    pub agents_executed: Vec<String>,

    /// Error message if failed.
    pub error: Option<String>,

    /// Quality contribution from this phase.
    pub quality_contribution: f64,
}

impl PhaseResult {
    /// Create a successful phase result.
    pub fn success(
        phase_name: impl Into<String>,
        duration_ms: u64,
        agents: Vec<String>,
        quality: f64,
    ) -> Self {
        Self {
            phase_name: phase_name.into(),
            success: true,
            duration_ms,
            agents_executed: agents,
            error: None,
            quality_contribution: quality,
        }
    }

    /// Create a failed phase result.
    pub fn failure(
        phase_name: impl Into<String>,
        duration_ms: u64,
        error: impl Into<String>,
    ) -> Self {
        Self {
            phase_name: phase_name.into(),
            success: false,
            duration_ms,
            agents_executed: Vec::new(),
            error: Some(error.into()),
            quality_contribution: 0.0,
        }
    }
}

impl TranscendentResponse {
    /// Create a new transcendent response.
    pub fn new(response: impl Into<String>, plan: OrchestrationPlan) -> Self {
        let now = Utc::now();
        Self {
            response: response.into(),
            orchestration_plan: plan,
            agents_used: Vec::new(),
            total_latency_ms: 0,
            quality_score: 0.0,
            phase_results: Vec::new(),
            warnings: Vec::new(),
            fallbacks_triggered: Vec::new(),
            started_at: now,
            completed_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Set the agents used.
    pub fn with_agents_used(mut self, agents: Vec<String>) -> Self {
        self.agents_used = agents;
        self
    }

    /// Set the total latency.
    pub fn with_latency_ms(mut self, ms: u64) -> Self {
        self.total_latency_ms = ms;
        self
    }

    /// Set the quality score.
    pub fn with_quality_score(mut self, score: f64) -> Self {
        self.quality_score = score.clamp(0.0, 1.0);
        self
    }

    /// Add a phase result.
    pub fn with_phase_result(mut self, result: PhaseResult) -> Self {
        self.phase_results.push(result);
        self
    }

    /// Add a warning.
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }

    /// Add a triggered fallback.
    pub fn with_fallback_triggered(mut self, fallback: impl Into<String>) -> Self {
        self.fallbacks_triggered.push(fallback.into());
        self
    }

    /// Set timestamps.
    pub fn with_timestamps(mut self, started: DateTime<Utc>, completed: DateTime<Utc>) -> Self {
        self.started_at = started;
        self.completed_at = completed;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get the number of successful phases.
    pub fn successful_phases(&self) -> usize {
        self.phase_results.iter().filter(|r| r.success).count()
    }

    /// Get the number of failed phases.
    pub fn failed_phases(&self) -> usize {
        self.phase_results.iter().filter(|r| !r.success).count()
    }

    /// Check if any fallbacks were triggered.
    pub fn had_fallbacks(&self) -> bool {
        !self.fallbacks_triggered.is_empty()
    }

    /// Check if there were any warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get a summary of the response.
    pub fn summary(&self) -> String {
        format!(
            "Transcendence: {} agents, {}ms, quality {:.2}, {} phases ({} success, {} failed)",
            self.agents_used.len(),
            self.total_latency_ms,
            self.quality_score,
            self.phase_results.len(),
            self.successful_phases(),
            self.failed_phases()
        )
    }
}

impl Default for TranscendentResponse {
    fn default() -> Self {
        Self::new("", OrchestrationPlan::default())
    }
}

// =============================================================================
// TRANSCENDER CONFIGURATION
// =============================================================================

/// Configuration for the Transcender Agent.
#[derive(Debug, Clone)]
pub struct TranscenderConfig {
    /// Default optimization settings.
    pub optimization_defaults: OptimizationTarget,

    /// Default timeout in milliseconds.
    pub timeout_ms: u64,

    /// Whether to enable parallel execution.
    pub parallel_execution: bool,

    /// Maximum concurrent agents.
    pub max_concurrent_agents: usize,

    /// Minimum quality threshold to accept a response.
    pub min_quality_threshold: f64,

    /// Maximum retries on failure.
    pub max_retries: u8,

    /// Whether to enable automatic fallbacks.
    pub enable_fallbacks: bool,

    /// Whether to enable adaptive optimization.
    pub adaptive_optimization: bool,

    /// Quality weight for combined scoring.
    pub quality_weight: f64,

    /// Speed weight for combined scoring.
    pub speed_weight: f64,

    /// Cost weight for combined scoring.
    pub cost_weight: f64,
}

impl Default for TranscenderConfig {
    fn default() -> Self {
        Self {
            optimization_defaults: OptimizationTarget::Balanced,
            timeout_ms: 30000,
            parallel_execution: true,
            max_concurrent_agents: 10,
            min_quality_threshold: 0.5,
            max_retries: 3,
            enable_fallbacks: true,
            adaptive_optimization: true,
            quality_weight: 0.5,
            speed_weight: 0.3,
            cost_weight: 0.2,
        }
    }
}

impl TranscenderConfig {
    /// Create a quality-focused configuration.
    pub fn quality_focused() -> Self {
        Self {
            optimization_defaults: OptimizationTarget::Quality,
            timeout_ms: 60000,
            parallel_execution: false,
            max_concurrent_agents: 5,
            min_quality_threshold: 0.8,
            max_retries: 5,
            enable_fallbacks: true,
            adaptive_optimization: true,
            quality_weight: 0.8,
            speed_weight: 0.1,
            cost_weight: 0.1,
        }
    }

    /// Create a speed-focused configuration.
    pub fn speed_focused() -> Self {
        Self {
            optimization_defaults: OptimizationTarget::Speed,
            timeout_ms: 10000,
            parallel_execution: true,
            max_concurrent_agents: 20,
            min_quality_threshold: 0.3,
            max_retries: 1,
            enable_fallbacks: false,
            adaptive_optimization: false,
            quality_weight: 0.2,
            speed_weight: 0.7,
            cost_weight: 0.1,
        }
    }

    /// Create a cost-focused configuration.
    pub fn cost_focused() -> Self {
        Self {
            optimization_defaults: OptimizationTarget::Cost,
            timeout_ms: 20000,
            parallel_execution: false,
            max_concurrent_agents: 3,
            min_quality_threshold: 0.4,
            max_retries: 2,
            enable_fallbacks: false,
            adaptive_optimization: false,
            quality_weight: 0.3,
            speed_weight: 0.2,
            cost_weight: 0.5,
        }
    }
}

// =============================================================================
// TRANSCENDER AGENT
// =============================================================================

/// The Transcender Agent - Agent 40 of Project Panpsychism.
///
/// The supreme orchestrator that coordinates all tiers with ultimate coordination,
/// tier management, and system evolution. Like a cosmic conductor, it harmonizes
/// the entire agent ecosystem into a unified, self-evolving system.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles at the highest level:
/// - **CONATUS**: The drive to optimize and evolve the entire system
/// - **NATURA**: Natural harmony between all agents and tiers
/// - **RATIO**: Supreme logical orchestration across all components
/// - **LAETITIA**: Joy through transcendent synthesis and emergence
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::transcender::{
///     TranscenderAgent, TranscendentGoal, OptimizationTarget
/// };
///
/// let transcender = TranscenderAgent::builder()
///     .parallel_execution(true)
///     .timeout_ms(30000)
///     .build();
///
/// let goal = TranscendentGoal::new("Generate a comprehensive analysis")
///     .with_optimization_target(OptimizationTarget::Quality);
///
/// let response = transcender.transcend(&goal).await?;
/// println!("Result: {}", response.response);
/// ```
#[derive(Debug, Clone)]
pub struct TranscenderAgent {
    /// Configuration for the agent.
    config: TranscenderConfig,
}

impl Default for TranscenderAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl TranscenderAgent {
    /// Create a new Transcender Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: TranscenderConfig::default(),
        }
    }

    /// Create a Transcender Agent with custom configuration.
    pub fn with_config(config: TranscenderConfig) -> Self {
        Self { config }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> TranscenderAgentBuilder {
        TranscenderAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &TranscenderConfig {
        &self.config
    }

    // =========================================================================
    // MAIN TRANSCENDENCE METHOD
    // =========================================================================

    /// Perform transcendent orchestration to achieve a goal.
    ///
    /// This is the primary method that orchestrates all agents optimally
    /// to achieve the specified goal.
    ///
    /// # Arguments
    ///
    /// * `goal` - The transcendent goal to achieve
    ///
    /// # Returns
    ///
    /// A `TranscendentResponse` containing the result and orchestration metadata.
    ///
    /// # Errors
    ///
    /// Returns `Error::Orchestration` if orchestration fails critically.
    pub async fn transcend(&self, goal: &TranscendentGoal) -> Result<TranscendentResponse> {
        let start_time = Instant::now();
        let started_at = Utc::now();

        info!(
            "Beginning transcendence for goal: {} (optimization: {})",
            goal.objective, goal.optimization_target
        );

        // Generate the orchestration plan
        let plan = self.generate_plan(goal)?;
        debug!("Generated plan: {}", plan.summary());

        // Execute the plan
        let mut response = self.execute_plan(&plan, goal).await?;

        // Calculate final quality score
        response.quality_score = self.calculate_quality_score(&response);

        // Set timing
        let completed_at = Utc::now();
        response.started_at = started_at;
        response.completed_at = completed_at;
        response.total_latency_ms = start_time.elapsed().as_millis() as u64;

        // Check quality threshold
        if response.quality_score < self.config.min_quality_threshold {
            response.warnings.push(format!(
                "Quality score {:.2} below threshold {:.2}",
                response.quality_score, self.config.min_quality_threshold
            ));
        }

        info!(
            "Transcendence complete: {}",
            response.summary()
        );

        Ok(response)
    }

    // =========================================================================
    // PLAN GENERATION
    // =========================================================================

    /// Generate an orchestration plan for a goal.
    pub fn generate_plan(&self, goal: &TranscendentGoal) -> Result<OrchestrationPlan> {
        let plan_id = format!("plan-{}", Utc::now().timestamp_millis());

        let mut plan = OrchestrationPlan::new(&plan_id);

        // Determine agents based on optimization target
        let agents = self.select_agents_for_goal(goal);

        // Create phases based on optimization strategy
        let phases = self.create_phases_for_goal(goal, &agents);
        plan.phases = phases;

        // Add parallel groups if enabled
        if self.config.parallel_execution {
            plan.parallel_groups = self.determine_parallel_groups(&plan.phases);
        }

        // Add fallback paths if enabled
        if self.config.enable_fallbacks {
            plan.fallback_paths = self.generate_fallback_paths(goal);
        }

        // Estimate metrics
        plan.estimated_duration_ms = self.estimate_duration(&plan);
        plan.estimated_quality = self.estimate_quality(&plan, goal);
        plan.estimated_cost = self.estimate_cost(&plan);

        plan.metadata.insert("optimization_target".to_string(), goal.optimization_target.to_string());
        plan.metadata.insert("time_budget_ms".to_string(), goal.time_budget_ms.to_string());

        Ok(plan)
    }

    /// Select agents appropriate for the goal.
    fn select_agents_for_goal(&self, goal: &TranscendentGoal) -> Vec<String> {
        let mut agents = Vec::new();

        // Base agents always included
        agents.push("analyzer".to_string());
        agents.push("synthesizer".to_string());

        // Add agents based on optimization target
        match &goal.optimization_target {
            OptimizationTarget::Quality => {
                agents.push("validator".to_string());
                agents.push("evaluator".to_string());
                agents.push("enhancer".to_string());
                agents.push("corrector".to_string());
            }
            OptimizationTarget::Speed => {
                agents.push("predictor".to_string());
            }
            OptimizationTarget::Cost => {
                // Minimal agents
            }
            OptimizationTarget::Balanced => {
                agents.push("validator".to_string());
                agents.push("predictor".to_string());
            }
            OptimizationTarget::Custom(weights) => {
                if weights.iter().any(|(n, w)| n == "quality" && *w > 0.5) {
                    agents.push("validator".to_string());
                    agents.push("enhancer".to_string());
                }
                if weights.iter().any(|(n, w)| n == "speed" && *w > 0.5) {
                    agents.push("predictor".to_string());
                }
            }
        }

        // Check constraint for required agents
        if let Some(constraint) = goal.get_constraint("required_agents") {
            for agent in constraint.as_list() {
                if !agents.contains(&agent) {
                    agents.push(agent);
                }
            }
        }

        // Check constraint for excluded agents
        if let Some(constraint) = goal.get_constraint("excluded_agents") {
            let excluded = constraint.as_list();
            agents.retain(|a| !excluded.contains(a));
        }

        agents
    }

    /// Create execution phases for a goal.
    fn create_phases_for_goal(&self, goal: &TranscendentGoal, agents: &[String]) -> Vec<Phase> {
        let mut phases = Vec::new();

        // Phase 1: Analysis
        let analysis_phase = Phase::new("analysis")
            .with_agents(agents.iter().filter(|a| a.contains("analyz")).cloned())
            .with_order(1)
            .with_timeout_ms(goal.time_budget_ms / 4)
            .with_description("Initial analysis and context gathering");
        phases.push(analysis_phase);

        // Phase 2: Synthesis
        let synthesis_phase = Phase::new("synthesis")
            .with_agents(agents.iter().filter(|a| a.contains("synth")).cloned())
            .with_order(2)
            .with_timeout_ms(goal.time_budget_ms / 3)
            .with_dependency("analysis")
            .with_description("Core synthesis and generation");
        phases.push(synthesis_phase);

        // Phase 3: Enhancement (if quality-focused)
        if goal.optimization_target.quality_weight() > 0.5 {
            let enhancement_phase = Phase::new("enhancement")
                .with_agents(agents.iter().filter(|a| {
                    a.contains("enhance") || a.contains("valid") || a.contains("eval")
                }).cloned())
                .with_order(3)
                .with_timeout_ms(goal.time_budget_ms / 4)
                .with_required(false)
                .with_dependency("synthesis")
                .with_description("Quality enhancement and validation");
            phases.push(enhancement_phase);
        }

        // Phase 4: Correction (optional)
        if agents.iter().any(|a| a.contains("correct")) {
            let correction_phase = Phase::new("correction")
                .with_agents(agents.iter().filter(|a| a.contains("correct")).cloned())
                .with_order(4)
                .with_timeout_ms(goal.time_budget_ms / 6)
                .with_required(false)
                .with_dependency("synthesis")
                .with_description("Error correction and refinement");
            phases.push(correction_phase);
        }

        phases
    }

    /// Determine which agents can run in parallel.
    fn determine_parallel_groups(&self, phases: &[Phase]) -> Vec<Vec<String>> {
        let mut parallel_groups = Vec::new();

        // Group independent phases
        for phase in phases {
            if !phase.has_dependencies() && phase.agents.len() > 1 {
                parallel_groups.push(phase.agents.clone());
            }
        }

        parallel_groups
    }

    /// Generate fallback paths for resilient execution.
    fn generate_fallback_paths(&self, _goal: &TranscendentGoal) -> Vec<FallbackPath> {
        vec![
            FallbackPath::on_timeout(&["fast_synthesizer", "cached_analyzer"]),
            FallbackPath::on_error(&["backup_analyzer", "simple_synthesizer"]),
            FallbackPath::on_low_quality(&["enhanced_validator", "quality_enhancer"]),
        ]
    }

    /// Estimate execution duration.
    fn estimate_duration(&self, plan: &OrchestrationPlan) -> u64 {
        if self.config.parallel_execution && !plan.parallel_groups.is_empty() {
            // Parallel: max of phase durations
            plan.phases.iter().map(|p| p.timeout_ms).max().unwrap_or(0)
        } else {
            // Sequential: sum of phase durations
            plan.phases.iter().map(|p| p.timeout_ms).sum()
        }
    }

    /// Estimate quality based on plan and goal.
    fn estimate_quality(&self, plan: &OrchestrationPlan, goal: &TranscendentGoal) -> f64 {
        let base_quality = 0.5;
        let agent_bonus = plan.unique_agents().len() as f64 * 0.05;
        let phase_bonus = plan.phases.len() as f64 * 0.05;

        let quality = base_quality + agent_bonus + phase_bonus;

        // Adjust based on optimization target
        let target_adjustment = match &goal.optimization_target {
            OptimizationTarget::Quality => 0.2,
            OptimizationTarget::Speed => -0.1,
            OptimizationTarget::Cost => -0.05,
            OptimizationTarget::Balanced => 0.0,
            OptimizationTarget::Custom(weights) => {
                weights.iter().find(|(n, _)| n == "quality")
                    .map(|(_, w)| w * 0.2)
                    .unwrap_or(0.0)
            }
        };

        (quality + target_adjustment).clamp(0.0, 1.0)
    }

    /// Estimate cost based on plan.
    fn estimate_cost(&self, plan: &OrchestrationPlan) -> f64 {
        let agent_cost = plan.unique_agents().len() as f64 * 1.0;
        let phase_cost = plan.phases.len() as f64 * 0.5;
        let duration_cost = plan.estimated_duration_ms as f64 / 1000.0 * 0.1;

        agent_cost + phase_cost + duration_cost
    }

    // =========================================================================
    // PLAN EXECUTION
    // =========================================================================

    /// Execute an orchestration plan.
    async fn execute_plan(
        &self,
        plan: &OrchestrationPlan,
        goal: &TranscendentGoal,
    ) -> Result<TranscendentResponse> {
        let mut response = TranscendentResponse::new("", plan.clone());
        let mut accumulated_results = Vec::new();

        // Execute phases in order
        let sorted_phases = plan.sorted_phases();

        for phase in sorted_phases {
            debug!("Executing phase: {}", phase.name);

            let phase_result = self.execute_phase(phase, goal).await;

            match phase_result {
                Ok(result) => {
                    accumulated_results.push(result.clone());
                    response.phase_results.push(result);
                }
                Err(e) => {
                    if phase.required {
                        // Required phase failed - try fallbacks
                        if self.config.enable_fallbacks {
                            if let Some(fallback_result) = self.try_fallbacks(plan, &phase.name).await {
                                response.fallbacks_triggered.push(phase.name.clone());
                                accumulated_results.push(fallback_result.clone());
                                response.phase_results.push(fallback_result);
                            } else {
                                return Err(Error::Orchestration(format!(
                                    "Required phase '{}' failed: {}",
                                    phase.name, e
                                )));
                            }
                        } else {
                            return Err(Error::Orchestration(format!(
                                "Required phase '{}' failed: {}",
                                phase.name, e
                            )));
                        }
                    } else {
                        // Optional phase failed - log warning
                        warn!("Optional phase '{}' failed: {}", phase.name, e);
                        response.warnings.push(format!(
                            "Optional phase '{}' failed: {}",
                            phase.name, e
                        ));
                        response.phase_results.push(PhaseResult::failure(
                            &phase.name,
                            0,
                            e.to_string(),
                        ));
                    }
                }
            }
        }

        // Synthesize final response from phase results
        response.response = self.synthesize_response(&accumulated_results);

        // Collect agents used
        response.agents_used = accumulated_results
            .iter()
            .flat_map(|r| r.agents_executed.clone())
            .collect();
        response.agents_used.sort();
        response.agents_used.dedup();

        Ok(response)
    }

    /// Execute a single phase.
    async fn execute_phase(
        &self,
        phase: &Phase,
        _goal: &TranscendentGoal,
    ) -> Result<PhaseResult> {
        let start = Instant::now();

        // Simulate agent execution
        // In a real implementation, this would call actual agents
        let agents_executed = phase.agents.clone();

        // Simulate processing time (in tests, this is instant)
        #[cfg(not(test))]
        {
            let delay = std::cmp::min(phase.timeout_ms / 10, 100);
            tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // Simulate quality contribution based on agent count
        let quality_contribution = 0.2 + (agents_executed.len() as f64 * 0.1).min(0.5);

        Ok(PhaseResult::success(
            &phase.name,
            duration_ms,
            agents_executed,
            quality_contribution,
        ))
    }

    /// Try fallback paths when a phase fails.
    async fn try_fallbacks(
        &self,
        plan: &OrchestrationPlan,
        _failed_phase: &str,
    ) -> Option<PhaseResult> {
        for fallback in &plan.fallback_paths {
            debug!("Trying fallback: {}", fallback.condition);

            // Simulate fallback execution
            if !fallback.alternative_agents.is_empty() {
                return Some(PhaseResult::success(
                    format!("fallback_{}", fallback.condition),
                    50,
                    fallback.alternative_agents.clone(),
                    0.3,
                ));
            }
        }

        None
    }

    /// Synthesize the final response from phase results.
    fn synthesize_response(&self, phase_results: &[PhaseResult]) -> String {
        let successful_phases: Vec<&PhaseResult> = phase_results
            .iter()
            .filter(|r| r.success)
            .collect();

        if successful_phases.is_empty() {
            return "Transcendence produced no results".to_string();
        }

        format!(
            "Transcendence complete: {} phases executed successfully with {} total agents",
            successful_phases.len(),
            successful_phases.iter().map(|r| r.agents_executed.len()).sum::<usize>()
        )
    }

    /// Calculate the final quality score.
    fn calculate_quality_score(&self, response: &TranscendentResponse) -> f64 {
        if response.phase_results.is_empty() {
            return 0.0;
        }

        let total_quality: f64 = response
            .phase_results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.quality_contribution)
            .sum();

        let success_rate = response.successful_phases() as f64 / response.phase_results.len() as f64;

        ((total_quality + success_rate) / 2.0).clamp(0.0, 1.0)
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Validate a goal before orchestration.
    pub fn validate_goal(&self, goal: &TranscendentGoal) -> Vec<String> {
        let mut errors = Vec::new();

        if goal.objective.trim().is_empty() {
            errors.push("Goal objective cannot be empty".to_string());
        }

        if goal.time_budget_ms == 0 {
            errors.push("Time budget must be greater than 0".to_string());
        }

        if goal.time_budget_ms > self.config.timeout_ms {
            errors.push(format!(
                "Goal time budget {}ms exceeds config timeout {}ms",
                goal.time_budget_ms, self.config.timeout_ms
            ));
        }

        // Validate constraints
        for constraint in &goal.constraints {
            if constraint.name.is_empty() {
                errors.push("Constraint name cannot be empty".to_string());
            }
            if constraint.value.is_empty() {
                errors.push(format!("Constraint '{}' has empty value", constraint.name));
            }
        }

        errors
    }

    /// Estimate resources needed for a goal.
    pub fn estimate_resources(&self, goal: &TranscendentGoal) -> HashMap<String, f64> {
        let plan = self.generate_plan(goal).unwrap_or_default();

        let mut resources = HashMap::new();
        resources.insert("agents".to_string(), plan.unique_agents().len() as f64);
        resources.insert("phases".to_string(), plan.phases.len() as f64);
        resources.insert("estimated_duration_ms".to_string(), plan.estimated_duration_ms as f64);
        resources.insert("estimated_quality".to_string(), plan.estimated_quality);
        resources.insert("estimated_cost".to_string(), plan.estimated_cost);

        resources
    }

    /// Check if the system can handle a goal.
    pub fn can_handle(&self, goal: &TranscendentGoal) -> bool {
        let errors = self.validate_goal(goal);
        errors.is_empty()
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom TranscenderAgent configuration.
#[derive(Debug, Default)]
pub struct TranscenderAgentBuilder {
    config: Option<TranscenderConfig>,
}

impl TranscenderAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: TranscenderConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the optimization defaults.
    pub fn optimization_defaults(mut self, target: OptimizationTarget) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.optimization_defaults = target;
        self.config = Some(config);
        self
    }

    /// Set the timeout in milliseconds.
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.timeout_ms = ms;
        self.config = Some(config);
        self
    }

    /// Set whether to enable parallel execution.
    pub fn parallel_execution(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.parallel_execution = enabled;
        self.config = Some(config);
        self
    }

    /// Set the maximum concurrent agents.
    pub fn max_concurrent_agents(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_concurrent_agents = max;
        self.config = Some(config);
        self
    }

    /// Set the minimum quality threshold.
    pub fn min_quality_threshold(mut self, threshold: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.min_quality_threshold = threshold.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set the maximum retries.
    pub fn max_retries(mut self, retries: u8) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_retries = retries;
        self.config = Some(config);
        self
    }

    /// Set whether to enable fallbacks.
    pub fn enable_fallbacks(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.enable_fallbacks = enabled;
        self.config = Some(config);
        self
    }

    /// Set whether to enable adaptive optimization.
    pub fn adaptive_optimization(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.adaptive_optimization = enabled;
        self.config = Some(config);
        self
    }

    /// Set the quality weight.
    pub fn quality_weight(mut self, weight: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.quality_weight = weight;
        self.config = Some(config);
        self
    }

    /// Set the speed weight.
    pub fn speed_weight(mut self, weight: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.speed_weight = weight;
        self.config = Some(config);
        self
    }

    /// Set the cost weight.
    pub fn cost_weight(mut self, weight: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.cost_weight = weight;
        self.config = Some(config);
        self
    }

    /// Build the TranscenderAgent.
    pub fn build(self) -> TranscenderAgent {
        TranscenderAgent {
            config: self.config.unwrap_or_default(),
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
    // OptimizationTarget Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimization_target_default() {
        assert_eq!(OptimizationTarget::default(), OptimizationTarget::Balanced);
    }

    #[test]
    fn test_optimization_target_display() {
        assert_eq!(OptimizationTarget::Quality.to_string(), "quality");
        assert_eq!(OptimizationTarget::Speed.to_string(), "speed");
        assert_eq!(OptimizationTarget::Cost.to_string(), "cost");
        assert_eq!(OptimizationTarget::Balanced.to_string(), "balanced");
    }

    #[test]
    fn test_optimization_target_custom_display() {
        let custom = OptimizationTarget::Custom(vec![
            ("quality".to_string(), 0.5),
            ("speed".to_string(), 0.3),
        ]);
        let display = custom.to_string();
        assert!(display.starts_with("custom("));
        assert!(display.contains("quality:0.50"));
        assert!(display.contains("speed:0.30"));
    }

    #[test]
    fn test_optimization_target_from_str() {
        assert_eq!(
            "quality".parse::<OptimizationTarget>().unwrap(),
            OptimizationTarget::Quality
        );
        assert_eq!(
            "fast".parse::<OptimizationTarget>().unwrap(),
            OptimizationTarget::Speed
        );
        assert_eq!(
            "cheap".parse::<OptimizationTarget>().unwrap(),
            OptimizationTarget::Cost
        );
        assert_eq!(
            "balanced".parse::<OptimizationTarget>().unwrap(),
            OptimizationTarget::Balanced
        );
    }

    #[test]
    fn test_optimization_target_from_str_invalid() {
        assert!("invalid".parse::<OptimizationTarget>().is_err());
    }

    #[test]
    fn test_optimization_target_weights() {
        assert_eq!(OptimizationTarget::Quality.quality_weight(), 1.0);
        assert_eq!(OptimizationTarget::Speed.speed_weight(), 1.0);
        assert_eq!(OptimizationTarget::Cost.cost_weight(), 1.0);
        assert_eq!(OptimizationTarget::Balanced.quality_weight(), 0.5);
    }

    #[test]
    fn test_optimization_target_custom_weights() {
        let custom = OptimizationTarget::custom(0.7, 0.2, 0.1);
        assert_eq!(custom.quality_weight(), 0.7);
        assert_eq!(custom.speed_weight(), 0.2);
        assert_eq!(custom.cost_weight(), 0.1);
    }

    #[test]
    fn test_optimization_target_all_standard() {
        let all = OptimizationTarget::all_standard();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&OptimizationTarget::Quality));
        assert!(all.contains(&OptimizationTarget::Speed));
    }

    // -------------------------------------------------------------------------
    // Constraint Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constraint_new() {
        let constraint = Constraint::new("max_latency", "number", "5000");
        assert_eq!(constraint.name, "max_latency");
        assert_eq!(constraint.constraint_type, "number");
        assert_eq!(constraint.value, "5000");
        assert!(constraint.strict);
    }

    #[test]
    fn test_constraint_builder() {
        let constraint = Constraint::new("test", "string", "value")
            .with_description("Test constraint")
            .with_strict(false);

        assert_eq!(constraint.description, Some("Test constraint".to_string()));
        assert!(!constraint.strict);
    }

    #[test]
    fn test_constraint_as_number() {
        let constraint = Constraint::new("limit", "number", "42.5");
        assert_eq!(constraint.as_number(), Some(42.5));

        let invalid = Constraint::new("limit", "number", "not_a_number");
        assert_eq!(invalid.as_number(), None);
    }

    #[test]
    fn test_constraint_as_boolean() {
        assert_eq!(
            Constraint::new("flag", "boolean", "true").as_boolean(),
            Some(true)
        );
        assert_eq!(
            Constraint::new("flag", "boolean", "yes").as_boolean(),
            Some(true)
        );
        assert_eq!(
            Constraint::new("flag", "boolean", "false").as_boolean(),
            Some(false)
        );
        assert_eq!(
            Constraint::new("flag", "boolean", "invalid").as_boolean(),
            None
        );
    }

    #[test]
    fn test_constraint_as_list() {
        let constraint = Constraint::new("agents", "list", "agent1, agent2, agent3");
        let list = constraint.as_list();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0], "agent1");
        assert_eq!(list[1], "agent2");
        assert_eq!(list[2], "agent3");
    }

    #[test]
    fn test_constraint_factory_methods() {
        let latency = Constraint::max_latency_ms(5000);
        assert_eq!(latency.name, "max_latency_ms");
        assert_eq!(latency.value, "5000");

        let quality = Constraint::min_quality(0.8);
        assert_eq!(quality.name, "min_quality");
        assert_eq!(quality.value, "0.8");

        let required = Constraint::required_agents(&["agent1", "agent2"]);
        assert_eq!(required.name, "required_agents");
        assert_eq!(required.value, "agent1,agent2");
    }

    #[test]
    fn test_constraint_default() {
        let constraint = Constraint::default();
        assert!(constraint.name.is_empty());
    }

    // -------------------------------------------------------------------------
    // TranscendentGoal Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transcendent_goal_new() {
        let goal = TranscendentGoal::new("Generate analysis");
        assert_eq!(goal.objective, "Generate analysis");
        assert!(goal.constraints.is_empty());
        assert_eq!(goal.optimization_target, OptimizationTarget::Balanced);
        assert_eq!(goal.time_budget_ms, 30000);
        assert_eq!(goal.priority, 5);
    }

    #[test]
    fn test_transcendent_goal_builder() {
        let goal = TranscendentGoal::new("Test")
            .with_constraint(Constraint::max_latency_ms(5000))
            .with_optimization_target(OptimizationTarget::Quality)
            .with_time_budget_ms(10000)
            .with_context("key", "value")
            .with_priority(8)
            .with_tag("important");

        assert_eq!(goal.constraints.len(), 1);
        assert_eq!(goal.optimization_target, OptimizationTarget::Quality);
        assert_eq!(goal.time_budget_ms, 10000);
        assert_eq!(goal.context.get("key"), Some(&"value".to_string()));
        assert_eq!(goal.priority, 8);
        assert!(goal.tags.contains(&"important".to_string()));
    }

    #[test]
    fn test_transcendent_goal_with_constraints() {
        let goal = TranscendentGoal::new("Test").with_constraints(vec![
            Constraint::max_latency_ms(5000),
            Constraint::min_quality(0.8),
        ]);

        assert_eq!(goal.constraints.len(), 2);
    }

    #[test]
    fn test_transcendent_goal_with_tags() {
        let goal = TranscendentGoal::new("Test")
            .with_tags(vec!["tag1", "tag2", "tag3"]);

        assert_eq!(goal.tags.len(), 3);
    }

    #[test]
    fn test_transcendent_goal_priority_clamped() {
        let goal = TranscendentGoal::new("Test").with_priority(15);
        assert_eq!(goal.priority, 10);
    }

    #[test]
    fn test_transcendent_goal_strict_constraints() {
        let goal = TranscendentGoal::new("Test")
            .with_constraint(Constraint::max_latency_ms(5000))
            .with_constraint(Constraint::min_quality(0.8).with_strict(false));

        let strict = goal.strict_constraints();
        let soft = goal.soft_constraints();

        assert_eq!(strict.len(), 1);
        assert_eq!(soft.len(), 1);
    }

    #[test]
    fn test_transcendent_goal_has_constraint() {
        let goal = TranscendentGoal::new("Test")
            .with_constraint(Constraint::max_latency_ms(5000));

        assert!(goal.has_constraint("max_latency_ms"));
        assert!(!goal.has_constraint("min_quality"));
    }

    #[test]
    fn test_transcendent_goal_get_constraint() {
        let goal = TranscendentGoal::new("Test")
            .with_constraint(Constraint::max_latency_ms(5000));

        let constraint = goal.get_constraint("max_latency_ms");
        assert!(constraint.is_some());
        assert_eq!(constraint.unwrap().value, "5000");
    }

    #[test]
    fn test_transcendent_goal_default() {
        let goal = TranscendentGoal::default();
        assert!(goal.objective.is_empty());
    }

    // -------------------------------------------------------------------------
    // Phase Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_phase_new() {
        let phase = Phase::new("analysis");
        assert_eq!(phase.name, "analysis");
        assert!(phase.agents.is_empty());
        assert_eq!(phase.timeout_ms, 5000);
        assert!(phase.required);
        assert_eq!(phase.order, 0);
    }

    #[test]
    fn test_phase_builder() {
        let phase = Phase::new("synthesis")
            .with_agent("synthesizer")
            .with_agents(vec!["validator", "enhancer"])
            .with_timeout_ms(10000)
            .with_required(false)
            .with_order(2)
            .with_description("Synthesis phase")
            .with_dependency("analysis");

        assert_eq!(phase.agents.len(), 3);
        assert_eq!(phase.timeout_ms, 10000);
        assert!(!phase.required);
        assert_eq!(phase.order, 2);
        assert_eq!(phase.description, Some("Synthesis phase".to_string()));
        assert!(phase.depends_on.contains(&"analysis".to_string()));
    }

    #[test]
    fn test_phase_has_dependencies() {
        let phase1 = Phase::new("phase1");
        let phase2 = Phase::new("phase2").with_dependency("phase1");

        assert!(!phase1.has_dependencies());
        assert!(phase2.has_dependencies());
    }

    #[test]
    fn test_phase_agent_count() {
        let phase = Phase::new("test")
            .with_agents(vec!["a", "b", "c"]);
        assert_eq!(phase.agent_count(), 3);
    }

    #[test]
    fn test_phase_default() {
        let phase = Phase::default();
        assert!(phase.name.is_empty());
    }

    // -------------------------------------------------------------------------
    // FallbackPath Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fallback_path_new() {
        let fallback = FallbackPath::new("agent_timeout");
        assert_eq!(fallback.condition, "agent_timeout");
        assert!(fallback.alternative_agents.is_empty());
        assert_eq!(fallback.priority, 5);
        assert_eq!(fallback.max_retries, 3);
    }

    #[test]
    fn test_fallback_path_builder() {
        let fallback = FallbackPath::new("error")
            .with_agent("backup_agent")
            .with_agents(vec!["agent1", "agent2"])
            .with_priority(10)
            .with_max_retries(5)
            .with_description("Fallback on error");

        assert_eq!(fallback.alternative_agents.len(), 3);
        assert_eq!(fallback.priority, 10);
        assert_eq!(fallback.max_retries, 5);
        assert!(fallback.description.is_some());
    }

    #[test]
    fn test_fallback_path_factory_methods() {
        let timeout = FallbackPath::on_timeout(&["fast_agent"]);
        assert_eq!(timeout.condition, "agent_timeout");

        let low_quality = FallbackPath::on_low_quality(&["quality_agent"]);
        assert_eq!(low_quality.condition, "quality_below_threshold");

        let error = FallbackPath::on_error(&["backup_agent"]);
        assert_eq!(error.condition, "agent_error");
    }

    #[test]
    fn test_fallback_path_default() {
        let fallback = FallbackPath::default();
        assert!(fallback.condition.is_empty());
    }

    // -------------------------------------------------------------------------
    // OrchestrationPlan Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_orchestration_plan_new() {
        let plan = OrchestrationPlan::new("plan-1");
        assert_eq!(plan.id, "plan-1");
        assert!(plan.phases.is_empty());
        assert!(plan.parallel_groups.is_empty());
        assert!(plan.fallback_paths.is_empty());
    }

    #[test]
    fn test_orchestration_plan_builder() {
        let plan = OrchestrationPlan::new("plan-1")
            .with_phase(Phase::new("analysis").with_agent("analyzer"))
            .with_parallel_group(vec!["a".to_string(), "b".to_string()])
            .with_fallback(FallbackPath::on_timeout(&["fast"]))
            .with_estimated_duration_ms(5000)
            .with_estimated_quality(0.8)
            .with_estimated_cost(10.0)
            .with_metadata("key", "value");

        assert_eq!(plan.phases.len(), 1);
        assert_eq!(plan.parallel_groups.len(), 1);
        assert_eq!(plan.fallback_paths.len(), 1);
        assert_eq!(plan.estimated_duration_ms, 5000);
        assert_eq!(plan.estimated_quality, 0.8);
        assert_eq!(plan.estimated_cost, 10.0);
    }

    #[test]
    fn test_orchestration_plan_with_phases() {
        let plan = OrchestrationPlan::new("plan-1")
            .with_phases(vec![
                Phase::new("phase1"),
                Phase::new("phase2"),
            ]);

        assert_eq!(plan.phases.len(), 2);
    }

    #[test]
    fn test_orchestration_plan_total_agent_count() {
        let plan = OrchestrationPlan::new("plan-1")
            .with_phase(Phase::new("p1").with_agents(vec!["a", "b"]))
            .with_phase(Phase::new("p2").with_agents(vec!["c"]));

        assert_eq!(plan.total_agent_count(), 3);
    }

    #[test]
    fn test_orchestration_plan_unique_agents() {
        let plan = OrchestrationPlan::new("plan-1")
            .with_phase(Phase::new("p1").with_agents(vec!["a", "b"]))
            .with_phase(Phase::new("p2").with_agents(vec!["b", "c"]));

        let unique = plan.unique_agents();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_orchestration_plan_required_optional_phases() {
        let plan = OrchestrationPlan::new("plan-1")
            .with_phase(Phase::new("required"))
            .with_phase(Phase::new("optional").with_required(false));

        assert_eq!(plan.required_phases().len(), 1);
        assert_eq!(plan.optional_phases().len(), 1);
    }

    #[test]
    fn test_orchestration_plan_sorted_phases() {
        let plan = OrchestrationPlan::new("plan-1")
            .with_phase(Phase::new("last").with_order(3))
            .with_phase(Phase::new("first").with_order(1))
            .with_phase(Phase::new("middle").with_order(2));

        let sorted = plan.sorted_phases();
        assert_eq!(sorted[0].name, "first");
        assert_eq!(sorted[1].name, "middle");
        assert_eq!(sorted[2].name, "last");
    }

    #[test]
    fn test_orchestration_plan_has_parallel_execution() {
        let plan1 = OrchestrationPlan::new("plan-1");
        let plan2 = OrchestrationPlan::new("plan-2")
            .with_parallel_group(vec!["a".to_string()]);

        assert!(!plan1.has_parallel_execution());
        assert!(plan2.has_parallel_execution());
    }

    #[test]
    fn test_orchestration_plan_summary() {
        let plan = OrchestrationPlan::new("plan-1")
            .with_phase(Phase::new("p1").with_agent("a"))
            .with_estimated_duration_ms(5000)
            .with_estimated_quality(0.8)
            .with_estimated_cost(10.0);

        let summary = plan.summary();
        assert!(summary.contains("plan-1"));
        assert!(summary.contains("1 phases"));
        assert!(summary.contains("5000ms"));
    }

    #[test]
    fn test_orchestration_plan_default() {
        let plan = OrchestrationPlan::default();
        assert!(plan.id.is_empty());
    }

    // -------------------------------------------------------------------------
    // PhaseResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_phase_result_success() {
        let result = PhaseResult::success(
            "analysis",
            100,
            vec!["analyzer".to_string()],
            0.5,
        );

        assert!(result.success);
        assert_eq!(result.phase_name, "analysis");
        assert_eq!(result.duration_ms, 100);
        assert_eq!(result.agents_executed.len(), 1);
        assert!(result.error.is_none());
        assert_eq!(result.quality_contribution, 0.5);
    }

    #[test]
    fn test_phase_result_failure() {
        let result = PhaseResult::failure("analysis", 50, "Timeout error");

        assert!(!result.success);
        assert_eq!(result.phase_name, "analysis");
        assert_eq!(result.duration_ms, 50);
        assert!(result.agents_executed.is_empty());
        assert_eq!(result.error, Some("Timeout error".to_string()));
        assert_eq!(result.quality_contribution, 0.0);
    }

    // -------------------------------------------------------------------------
    // TranscendentResponse Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transcendent_response_new() {
        let plan = OrchestrationPlan::new("plan-1");
        let response = TranscendentResponse::new("Result", plan);

        assert_eq!(response.response, "Result");
        assert!(response.agents_used.is_empty());
        assert_eq!(response.total_latency_ms, 0);
        assert_eq!(response.quality_score, 0.0);
    }

    #[test]
    fn test_transcendent_response_builder() {
        let plan = OrchestrationPlan::new("plan-1");
        let response = TranscendentResponse::new("Result", plan)
            .with_agents_used(vec!["a".to_string(), "b".to_string()])
            .with_latency_ms(1000)
            .with_quality_score(0.85)
            .with_phase_result(PhaseResult::success("p1", 100, vec![], 0.5))
            .with_warning("Warning 1")
            .with_fallback_triggered("fallback1")
            .with_metadata("key", "value");

        assert_eq!(response.agents_used.len(), 2);
        assert_eq!(response.total_latency_ms, 1000);
        assert_eq!(response.quality_score, 0.85);
        assert_eq!(response.phase_results.len(), 1);
        assert_eq!(response.warnings.len(), 1);
        assert_eq!(response.fallbacks_triggered.len(), 1);
    }

    #[test]
    fn test_transcendent_response_quality_clamped() {
        let plan = OrchestrationPlan::new("plan-1");
        let response = TranscendentResponse::new("Result", plan)
            .with_quality_score(1.5);

        assert_eq!(response.quality_score, 1.0);
    }

    #[test]
    fn test_transcendent_response_successful_failed_phases() {
        let plan = OrchestrationPlan::new("plan-1");
        let response = TranscendentResponse::new("Result", plan)
            .with_phase_result(PhaseResult::success("p1", 100, vec![], 0.5))
            .with_phase_result(PhaseResult::failure("p2", 50, "Error"))
            .with_phase_result(PhaseResult::success("p3", 75, vec![], 0.3));

        assert_eq!(response.successful_phases(), 2);
        assert_eq!(response.failed_phases(), 1);
    }

    #[test]
    fn test_transcendent_response_had_fallbacks() {
        let plan = OrchestrationPlan::new("plan-1");
        let response1 = TranscendentResponse::new("Result", plan.clone());
        let response2 = TranscendentResponse::new("Result", plan)
            .with_fallback_triggered("fallback1");

        assert!(!response1.had_fallbacks());
        assert!(response2.had_fallbacks());
    }

    #[test]
    fn test_transcendent_response_has_warnings() {
        let plan = OrchestrationPlan::new("plan-1");
        let response1 = TranscendentResponse::new("Result", plan.clone());
        let response2 = TranscendentResponse::new("Result", plan)
            .with_warning("Warning");

        assert!(!response1.has_warnings());
        assert!(response2.has_warnings());
    }

    #[test]
    fn test_transcendent_response_summary() {
        let plan = OrchestrationPlan::new("plan-1");
        let response = TranscendentResponse::new("Result", plan)
            .with_agents_used(vec!["a".to_string()])
            .with_latency_ms(1000)
            .with_quality_score(0.8)
            .with_phase_result(PhaseResult::success("p1", 100, vec![], 0.5));

        let summary = response.summary();
        assert!(summary.contains("1 agents"));
        assert!(summary.contains("1000ms"));
        assert!(summary.contains("0.80"));
    }

    #[test]
    fn test_transcendent_response_default() {
        let response = TranscendentResponse::default();
        assert!(response.response.is_empty());
    }

    // -------------------------------------------------------------------------
    // TranscenderConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transcender_config_default() {
        let config = TranscenderConfig::default();
        assert_eq!(config.optimization_defaults, OptimizationTarget::Balanced);
        assert_eq!(config.timeout_ms, 30000);
        assert!(config.parallel_execution);
        assert_eq!(config.max_concurrent_agents, 10);
        assert_eq!(config.min_quality_threshold, 0.5);
    }

    #[test]
    fn test_transcender_config_quality_focused() {
        let config = TranscenderConfig::quality_focused();
        assert_eq!(config.optimization_defaults, OptimizationTarget::Quality);
        assert_eq!(config.timeout_ms, 60000);
        assert!(!config.parallel_execution);
        assert_eq!(config.min_quality_threshold, 0.8);
    }

    #[test]
    fn test_transcender_config_speed_focused() {
        let config = TranscenderConfig::speed_focused();
        assert_eq!(config.optimization_defaults, OptimizationTarget::Speed);
        assert_eq!(config.timeout_ms, 10000);
        assert!(config.parallel_execution);
        assert_eq!(config.max_concurrent_agents, 20);
    }

    #[test]
    fn test_transcender_config_cost_focused() {
        let config = TranscenderConfig::cost_focused();
        assert_eq!(config.optimization_defaults, OptimizationTarget::Cost);
        assert!(!config.parallel_execution);
        assert_eq!(config.max_concurrent_agents, 3);
    }

    // -------------------------------------------------------------------------
    // TranscenderAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transcender_agent_new() {
        let agent = TranscenderAgent::new();
        assert_eq!(
            agent.config().optimization_defaults,
            OptimizationTarget::Balanced
        );
    }

    #[test]
    fn test_transcender_agent_with_config() {
        let config = TranscenderConfig::quality_focused();
        let agent = TranscenderAgent::with_config(config);
        assert_eq!(
            agent.config().optimization_defaults,
            OptimizationTarget::Quality
        );
    }

    #[test]
    fn test_transcender_agent_builder() {
        let agent = TranscenderAgent::builder()
            .optimization_defaults(OptimizationTarget::Speed)
            .timeout_ms(15000)
            .parallel_execution(false)
            .max_concurrent_agents(5)
            .min_quality_threshold(0.7)
            .max_retries(5)
            .enable_fallbacks(false)
            .adaptive_optimization(false)
            .quality_weight(0.6)
            .speed_weight(0.3)
            .cost_weight(0.1)
            .build();

        assert_eq!(
            agent.config().optimization_defaults,
            OptimizationTarget::Speed
        );
        assert_eq!(agent.config().timeout_ms, 15000);
        assert!(!agent.config().parallel_execution);
        assert_eq!(agent.config().max_concurrent_agents, 5);
        assert_eq!(agent.config().min_quality_threshold, 0.7);
        assert_eq!(agent.config().max_retries, 5);
        assert!(!agent.config().enable_fallbacks);
        assert!(!agent.config().adaptive_optimization);
    }

    #[test]
    fn test_transcender_agent_default() {
        let agent = TranscenderAgent::default();
        assert_eq!(
            agent.config().optimization_defaults,
            OptimizationTarget::Balanced
        );
    }

    // -------------------------------------------------------------------------
    // Goal Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_goal_empty_objective() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("");

        let errors = agent.validate_goal(&goal);
        assert!(errors.iter().any(|e| e.contains("objective")));
    }

    #[test]
    fn test_validate_goal_zero_time_budget() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Test").with_time_budget_ms(0);

        let errors = agent.validate_goal(&goal);
        assert!(errors.iter().any(|e| e.contains("Time budget")));
    }

    #[test]
    fn test_validate_goal_exceeds_timeout() {
        let agent = TranscenderAgent::builder()
            .timeout_ms(10000)
            .build();
        let goal = TranscendentGoal::new("Test").with_time_budget_ms(20000);

        let errors = agent.validate_goal(&goal);
        assert!(errors.iter().any(|e| e.contains("exceeds")));
    }

    #[test]
    fn test_validate_goal_empty_constraint() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Test")
            .with_constraint(Constraint::new("", "string", "value"));

        let errors = agent.validate_goal(&goal);
        assert!(errors.iter().any(|e| e.contains("Constraint name")));
    }

    #[test]
    fn test_validate_goal_valid() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Generate analysis")
            .with_time_budget_ms(5000)
            .with_constraint(Constraint::max_latency_ms(3000));

        let errors = agent.validate_goal(&goal);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_can_handle() {
        let agent = TranscenderAgent::new();

        let valid_goal = TranscendentGoal::new("Test").with_time_budget_ms(5000);
        let invalid_goal = TranscendentGoal::new("");

        assert!(agent.can_handle(&valid_goal));
        assert!(!agent.can_handle(&invalid_goal));
    }

    // -------------------------------------------------------------------------
    // Plan Generation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_plan_basic() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Test")
            .with_time_budget_ms(10000);

        let plan = agent.generate_plan(&goal).unwrap();

        assert!(!plan.id.is_empty());
        assert!(!plan.phases.is_empty());
        assert!(plan.estimated_duration_ms > 0);
        assert!(plan.estimated_quality > 0.0);
    }

    #[test]
    fn test_generate_plan_quality_optimization() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Test")
            .with_optimization_target(OptimizationTarget::Quality)
            .with_time_budget_ms(10000);

        let plan = agent.generate_plan(&goal).unwrap();

        // Quality optimization should include enhancement phase
        let has_enhancement = plan.phases.iter().any(|p| p.name == "enhancement");
        assert!(has_enhancement);
    }

    #[test]
    fn test_generate_plan_with_required_agents() {
        let agent = TranscenderAgent::new();
        // Use an agent name that matches a phase filter pattern (contains "analyzer")
        let goal = TranscendentGoal::new("Test")
            .with_constraint(Constraint::required_agents(&["custom_analyzer"]))
            .with_time_budget_ms(10000);

        let plan = agent.generate_plan(&goal).unwrap();

        let all_agents = plan.unique_agents();
        // The custom_analyzer matches the "analyz" filter in the analysis phase
        assert!(all_agents.contains(&"custom_analyzer".to_string()));
    }

    #[test]
    fn test_generate_plan_with_excluded_agents() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Test")
            .with_constraint(Constraint::excluded_agents(&["analyzer"]))
            .with_time_budget_ms(10000);

        let plan = agent.generate_plan(&goal).unwrap();

        let all_agents = plan.unique_agents();
        assert!(!all_agents.contains(&"analyzer".to_string()));
    }

    #[test]
    fn test_generate_plan_parallel_groups() {
        let agent = TranscenderAgent::builder()
            .parallel_execution(true)
            .build();
        let goal = TranscendentGoal::new("Test")
            .with_time_budget_ms(10000);

        let plan = agent.generate_plan(&goal).unwrap();

        // Plan should have metadata about optimization
        assert!(plan.metadata.contains_key("optimization_target"));
    }

    #[test]
    fn test_generate_plan_fallback_paths() {
        let agent = TranscenderAgent::builder()
            .enable_fallbacks(true)
            .build();
        let goal = TranscendentGoal::new("Test")
            .with_time_budget_ms(10000);

        let plan = agent.generate_plan(&goal).unwrap();

        assert!(!plan.fallback_paths.is_empty());
    }

    #[test]
    fn test_generate_plan_no_fallbacks() {
        let agent = TranscenderAgent::builder()
            .enable_fallbacks(false)
            .build();
        let goal = TranscendentGoal::new("Test")
            .with_time_budget_ms(10000);

        let plan = agent.generate_plan(&goal).unwrap();

        assert!(plan.fallback_paths.is_empty());
    }

    // -------------------------------------------------------------------------
    // Resource Estimation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_estimate_resources() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Test")
            .with_time_budget_ms(10000);

        let resources = agent.estimate_resources(&goal);

        assert!(resources.contains_key("agents"));
        assert!(resources.contains_key("phases"));
        assert!(resources.contains_key("estimated_duration_ms"));
        assert!(resources.contains_key("estimated_quality"));
        assert!(resources.contains_key("estimated_cost"));
    }

    // -------------------------------------------------------------------------
    // Transcendence Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_transcend_basic() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Generate analysis")
            .with_time_budget_ms(10000);

        let response = agent.transcend(&goal).await.unwrap();

        assert!(!response.response.is_empty());
        assert!(!response.agents_used.is_empty());
        assert!(response.quality_score > 0.0);
    }

    #[tokio::test]
    async fn test_transcend_quality_optimization() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Generate analysis")
            .with_optimization_target(OptimizationTarget::Quality)
            .with_time_budget_ms(10000);

        let response = agent.transcend(&goal).await.unwrap();

        assert!(!response.phase_results.is_empty());
        assert!(response.successful_phases() > 0);
    }

    #[tokio::test]
    async fn test_transcend_speed_optimization() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Quick task")
            .with_optimization_target(OptimizationTarget::Speed)
            .with_time_budget_ms(5000);

        let response = agent.transcend(&goal).await.unwrap();

        assert!(!response.response.is_empty());
    }

    #[tokio::test]
    async fn test_transcend_with_constraints() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Constrained task")
            .with_constraint(Constraint::max_latency_ms(5000))
            .with_constraint(Constraint::min_quality(0.6))
            .with_time_budget_ms(10000);

        let response = agent.transcend(&goal).await.unwrap();

        assert!(!response.response.is_empty());
    }

    #[tokio::test]
    async fn test_transcend_tracks_timing() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Test")
            .with_time_budget_ms(10000);

        let response = agent.transcend(&goal).await.unwrap();

        assert!(response.started_at <= response.completed_at);
    }

    #[tokio::test]
    async fn test_transcend_low_quality_warning() {
        let agent = TranscenderAgent::builder()
            .min_quality_threshold(0.99)
            .build();
        let goal = TranscendentGoal::new("Test")
            .with_time_budget_ms(10000);

        let response = agent.transcend(&goal).await.unwrap();

        // Should have warning about quality below threshold
        assert!(response.has_warnings());
        assert!(response.warnings.iter().any(|w| w.contains("below threshold")));
    }

    #[tokio::test]
    async fn test_transcend_records_phase_results() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Test")
            .with_time_budget_ms(10000);

        let response = agent.transcend(&goal).await.unwrap();

        assert!(!response.phase_results.is_empty());
        for result in &response.phase_results {
            assert!(!result.phase_name.is_empty());
        }
    }

    #[tokio::test]
    async fn test_transcend_custom_optimization() {
        let agent = TranscenderAgent::new();
        let goal = TranscendentGoal::new("Custom optimization")
            .with_optimization_target(OptimizationTarget::custom(0.6, 0.3, 0.1))
            .with_time_budget_ms(10000);

        let response = agent.transcend(&goal).await.unwrap();

        assert!(!response.response.is_empty());
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let builder = TranscenderAgentBuilder::default();
        let agent = builder.build();
        assert_eq!(
            agent.config().optimization_defaults,
            OptimizationTarget::Balanced
        );
    }

    #[test]
    fn test_builder_config() {
        let config = TranscenderConfig::speed_focused();
        let agent = TranscenderAgentBuilder::default()
            .config(config)
            .build();
        assert_eq!(
            agent.config().optimization_defaults,
            OptimizationTarget::Speed
        );
    }

    #[test]
    fn test_builder_chain() {
        let agent = TranscenderAgentBuilder::default()
            .optimization_defaults(OptimizationTarget::Quality)
            .timeout_ms(60000)
            .parallel_execution(false)
            .max_concurrent_agents(5)
            .min_quality_threshold(0.9)
            .max_retries(10)
            .enable_fallbacks(true)
            .adaptive_optimization(true)
            .quality_weight(0.8)
            .speed_weight(0.1)
            .cost_weight(0.1)
            .build();

        assert_eq!(
            agent.config().optimization_defaults,
            OptimizationTarget::Quality
        );
        assert_eq!(agent.config().timeout_ms, 60000);
        assert!(!agent.config().parallel_execution);
        assert_eq!(agent.config().max_concurrent_agents, 5);
        assert_eq!(agent.config().min_quality_threshold, 0.9);
        assert_eq!(agent.config().max_retries, 10);
        assert!(agent.config().enable_fallbacks);
        assert!(agent.config().adaptive_optimization);
        assert_eq!(agent.config().quality_weight, 0.8);
        assert_eq!(agent.config().speed_weight, 0.1);
        assert_eq!(agent.config().cost_weight, 0.1);
    }

    #[test]
    fn test_builder_min_quality_threshold_clamped() {
        let agent = TranscenderAgent::builder()
            .min_quality_threshold(1.5)
            .build();
        assert_eq!(agent.config().min_quality_threshold, 1.0);
    }
}
