//! Debugger Agent module for Project Panpsychism.
//!
//! Implements "The Error Hunter" — an agent that identifies issues, analyzes
//! root causes, and suggests fixes. Like a master detective in the arcane arts,
//! the Debugger Agent traces the threads of causality to uncover the source of
//! problems and illuminate the path to resolution.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In the mystical realm of spellcraft, errors are like curses that corrupt
//! the flow of magic. The Debugger Agent serves as the **Error Hunter** —
//! a specialist in the art of curse-breaking:
//!
//! - **Issues** are the manifestations of corrupted magic
//! - **Root Causes** are the sources of the corruption
//! - **Fixes** are the counter-spells that restore harmony
//! - **Prevention Tips** are the wards that protect against future curses
//!
//! ## Philosophy
//!
//! Grounded in Spinoza's principles:
//!
//! - **CONATUS**: Drive to restore system integrity and optimal function
//! - **RATIO**: Logical analysis tracing effects back to causes
//! - **LAETITIA**: Joy through resolution and restored functionality
//! - **NATURA**: Natural understanding of error propagation patterns
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::debugger::{DebuggerAgent, Issue, IssueCategory, Severity};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let agent = DebuggerAgent::new();
//!
//!     let issue = Issue::new(
//!         IssueCategory::ValidationFailed,
//!         Severity::High,
//!         "Spinoza validation failed for CONATUS principle",
//!     );
//!
//!     let report = agent.debug(&issue).await?;
//!     println!("Root cause: {}", report.root_cause.description);
//!     for fix in &report.suggested_fixes {
//!         println!("Fix: {} (confidence: {:.0}%)", fix.description, fix.confidence * 100.0);
//!     }
//!     Ok(())
//! }
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info};

// =============================================================================
// ISSUE CATEGORY
// =============================================================================

/// Categories of issues that the Debugger Agent can analyze.
///
/// Each category represents a different type of problem that may occur
/// in the Panpsychism system, from malformed queries to resource exhaustion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueCategory {
    /// The query is malformed or invalid.
    ///
    /// Examples: Empty query, invalid syntax, unsupported characters.
    QueryMalformed,

    /// The requested prompt was not found.
    ///
    /// Examples: Missing file, invalid ID, deleted prompt.
    PromptNotFound,

    /// Validation failed (Spinoza principles not met).
    ///
    /// Examples: CONATUS score too low, RATIO check failed.
    ValidationFailed,

    /// Synthesis of prompts failed.
    ///
    /// Examples: Template error, context overflow, LLM failure.
    SynthesisFailed,

    /// Operation exceeded time limit.
    ///
    /// Examples: API timeout, LLM response too slow.
    TimeoutExceeded,

    /// System resources are exhausted.
    ///
    /// Examples: Memory limit, rate limit, disk full.
    ResourceExhausted,

    /// Configuration is invalid or missing.
    ///
    /// Examples: Missing API key, invalid endpoint, bad config file.
    ConfigurationError,

    /// Network communication failed.
    ///
    /// Examples: Connection refused, DNS failure, SSL error.
    NetworkError,
}

impl std::fmt::Display for IssueCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueryMalformed => write!(f, "Query Malformed"),
            Self::PromptNotFound => write!(f, "Prompt Not Found"),
            Self::ValidationFailed => write!(f, "Validation Failed"),
            Self::SynthesisFailed => write!(f, "Synthesis Failed"),
            Self::TimeoutExceeded => write!(f, "Timeout Exceeded"),
            Self::ResourceExhausted => write!(f, "Resource Exhausted"),
            Self::ConfigurationError => write!(f, "Configuration Error"),
            Self::NetworkError => write!(f, "Network Error"),
        }
    }
}

impl IssueCategory {
    /// Get all issue categories.
    pub fn all() -> Vec<Self> {
        vec![
            Self::QueryMalformed,
            Self::PromptNotFound,
            Self::ValidationFailed,
            Self::SynthesisFailed,
            Self::TimeoutExceeded,
            Self::ResourceExhausted,
            Self::ConfigurationError,
            Self::NetworkError,
        ]
    }

    /// Get the typical root cause patterns for this category.
    pub fn typical_causes(&self) -> Vec<&'static str> {
        match self {
            Self::QueryMalformed => vec![
                "Empty or whitespace-only query",
                "Invalid characters in query",
                "Query exceeds maximum length",
                "Unsupported query format",
            ],
            Self::PromptNotFound => vec![
                "Prompt file was deleted or moved",
                "Invalid prompt ID format",
                "Index is out of date",
                "Permission denied to access prompt",
            ],
            Self::ValidationFailed => vec![
                "Content lacks clarity (RATIO principle)",
                "No clear purpose detected (CONATUS principle)",
                "Negative sentiment detected (LAETITIA principle)",
                "Unnatural language patterns (NATURA principle)",
            ],
            Self::SynthesisFailed => vec![
                "Template variable not found",
                "Context window exceeded",
                "LLM returned empty response",
                "Incompatible prompt versions",
            ],
            Self::TimeoutExceeded => vec![
                "API endpoint is slow or overloaded",
                "Network latency is high",
                "Request is too complex",
                "Server is under heavy load",
            ],
            Self::ResourceExhausted => vec![
                "Rate limit exceeded",
                "Memory allocation failed",
                "Disk space exhausted",
                "Too many concurrent requests",
            ],
            Self::ConfigurationError => vec![
                "API key is missing or invalid",
                "Endpoint URL is malformed",
                "Required configuration file not found",
                "Environment variable not set",
            ],
            Self::NetworkError => vec![
                "Connection refused by server",
                "DNS resolution failed",
                "SSL/TLS handshake failed",
                "Network is unreachable",
            ],
        }
    }

    /// Get the recommended error code for this category.
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::QueryMalformed => "E020",
            Self::PromptNotFound => "E010",
            Self::ValidationFailed => "E040",
            Self::SynthesisFailed => "E050",
            Self::TimeoutExceeded => "E031",
            Self::ResourceExhausted => "E034",
            Self::ConfigurationError => "E001",
            Self::NetworkError => "E030",
        }
    }
}

impl std::str::FromStr for IssueCategory {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "query_malformed" | "querymalformed" | "malformed" => Ok(Self::QueryMalformed),
            "prompt_not_found" | "promptnotfound" | "notfound" => Ok(Self::PromptNotFound),
            "validation_failed" | "validationfailed" | "validation" => Ok(Self::ValidationFailed),
            "synthesis_failed" | "synthesisfailed" | "synthesis" => Ok(Self::SynthesisFailed),
            "timeout_exceeded" | "timeoutexceeded" | "timeout" => Ok(Self::TimeoutExceeded),
            "resource_exhausted" | "resourceexhausted" | "resource" => Ok(Self::ResourceExhausted),
            "configuration_error" | "configurationerror" | "config" => Ok(Self::ConfigurationError),
            "network_error" | "networkerror" | "network" => Ok(Self::NetworkError),
            _ => Err(Error::Config(format!(
                "Unknown issue category: '{}'. Valid categories: query_malformed, prompt_not_found, \
                 validation_failed, synthesis_failed, timeout_exceeded, resource_exhausted, \
                 configuration_error, network_error",
                s
            ))),
        }
    }
}

// =============================================================================
// SEVERITY
// =============================================================================

/// Severity levels for issues.
///
/// Severity indicates the urgency and impact of an issue,
/// guiding prioritization of debugging efforts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Low severity — minor inconvenience, workaround available.
    Low,
    /// Medium severity — noticeable impact, should be addressed soon.
    Medium,
    /// High severity — significant impact, requires prompt attention.
    High,
    /// Critical severity — system failure, requires immediate action.
    Critical,
}

impl Default for Severity {
    fn default() -> Self {
        Self::Medium
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

impl Severity {
    /// Get all severity levels in order from lowest to highest.
    pub fn all() -> Vec<Self> {
        vec![Self::Low, Self::Medium, Self::High, Self::Critical]
    }

    /// Check if this severity requires immediate attention.
    pub fn is_urgent(&self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }

    /// Get a numeric priority value (higher = more urgent).
    pub fn priority(&self) -> u8 {
        match self {
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }

    /// Get the recommended response time for this severity.
    pub fn response_time(&self) -> &'static str {
        match self {
            Self::Low => "When convenient",
            Self::Medium => "Within 24 hours",
            Self::High => "Within 4 hours",
            Self::Critical => "Immediately",
        }
    }
}

impl std::str::FromStr for Severity {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" | "l" | "1" => Ok(Self::Low),
            "medium" | "med" | "m" | "2" => Ok(Self::Medium),
            "high" | "h" | "3" => Ok(Self::High),
            "critical" | "crit" | "c" | "4" => Ok(Self::Critical),
            _ => Err(Error::Config(format!(
                "Unknown severity: '{}'. Valid severities: low, medium, high, critical",
                s
            ))),
        }
    }
}

// =============================================================================
// COMPLEXITY
// =============================================================================

/// Complexity level for fixes.
///
/// Indicates the effort required to implement a fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Complexity {
    /// Simple fix — quick change, minimal risk.
    Simple,
    /// Moderate fix — some effort required, manageable risk.
    #[default]
    Moderate,
    /// Complex fix — significant effort, requires careful planning.
    Complex,
}

impl std::fmt::Display for Complexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Simple => write!(f, "Simple"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Complex => write!(f, "Complex"),
        }
    }
}

impl Complexity {
    /// Get all complexity levels.
    pub fn all() -> Vec<Self> {
        vec![Self::Simple, Self::Moderate, Self::Complex]
    }

    /// Get the estimated time to implement for this complexity.
    pub fn estimated_time(&self) -> &'static str {
        match self {
            Self::Simple => "< 30 minutes",
            Self::Moderate => "1-4 hours",
            Self::Complex => "> 4 hours",
        }
    }

    /// Get a numeric effort value (higher = more effort).
    pub fn effort(&self) -> u8 {
        match self {
            Self::Simple => 1,
            Self::Moderate => 2,
            Self::Complex => 3,
        }
    }
}

impl std::str::FromStr for Complexity {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "simple" | "easy" | "quick" | "1" => Ok(Self::Simple),
            "moderate" | "medium" | "mod" | "2" => Ok(Self::Moderate),
            "complex" | "hard" | "difficult" | "3" => Ok(Self::Complex),
            _ => Err(Error::Config(format!(
                "Unknown complexity: '{}'. Valid complexities: simple, moderate, complex",
                s
            ))),
        }
    }
}

// =============================================================================
// ISSUE
// =============================================================================

/// An issue identified by the Debugger Agent.
///
/// Issues are the manifestations of problems in the system,
/// each carrying metadata about its category, severity, and context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    /// The category of the issue.
    pub category: IssueCategory,
    /// The severity of the issue.
    pub severity: Severity,
    /// A description of the issue.
    pub description: String,
    /// The location where the issue occurred (file path, function, etc.).
    pub location: Option<String>,
    /// Additional context about the issue.
    pub context: Option<String>,
}

impl Issue {
    /// Create a new issue.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let issue = Issue::new(
    ///     IssueCategory::ValidationFailed,
    ///     Severity::High,
    ///     "CONATUS score 0.3 below threshold 0.5",
    /// );
    /// ```
    pub fn new(
        category: IssueCategory,
        severity: Severity,
        description: impl Into<String>,
    ) -> Self {
        Self {
            category,
            severity,
            description: description.into(),
            location: None,
            context: None,
        }
    }

    /// Set the location.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Set additional context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Check if this issue is urgent.
    pub fn is_urgent(&self) -> bool {
        self.severity.is_urgent()
    }

    /// Get the error code for this issue.
    pub fn error_code(&self) -> &'static str {
        self.category.error_code()
    }

    /// Format the issue as a summary string.
    pub fn summary(&self) -> String {
        let location = self
            .location
            .as_ref()
            .map(|l| format!(" at {}", l))
            .unwrap_or_default();
        format!(
            "[{}] {} ({}): {}{}",
            self.category.error_code(),
            self.category,
            self.severity,
            self.description,
            location
        )
    }
}

// =============================================================================
// ROOT CAUSE
// =============================================================================

/// The root cause of an issue.
///
/// Root causes trace the issue back to its source,
/// providing evidence and confidence in the analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCause {
    /// Description of the root cause.
    pub description: String,
    /// Confidence in this root cause analysis (0.0 to 1.0).
    pub confidence: f64,
    /// Evidence supporting this root cause.
    pub evidence: Vec<String>,
}

impl RootCause {
    /// Create a new root cause.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let cause = RootCause::new("API key expired", 0.95)
    ///     .with_evidence(vec!["401 response received", "Token timestamp expired"]);
    /// ```
    pub fn new(description: impl Into<String>, confidence: f64) -> Self {
        Self {
            description: description.into(),
            confidence: confidence.clamp(0.0, 1.0),
            evidence: Vec::new(),
        }
    }

    /// Add evidence to support this root cause.
    pub fn with_evidence(mut self, evidence: Vec<String>) -> Self {
        self.evidence = evidence;
        self
    }

    /// Add a single piece of evidence.
    pub fn add_evidence(&mut self, evidence: impl Into<String>) {
        self.evidence.push(evidence.into());
    }

    /// Check if this is a high-confidence analysis.
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Check if there is supporting evidence.
    pub fn has_evidence(&self) -> bool {
        !self.evidence.is_empty()
    }

    /// Format the root cause as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!(
            "## Root Cause Analysis\n\n**Cause:** {}\n\n**Confidence:** {:.0}%\n",
            self.description,
            self.confidence * 100.0
        );

        if !self.evidence.is_empty() {
            output.push_str("\n### Evidence\n\n");
            for e in &self.evidence {
                output.push_str(&format!("- {}\n", e));
            }
        }

        output
    }
}

// =============================================================================
// FIX
// =============================================================================

/// A suggested fix for an issue.
///
/// Fixes are counter-spells that restore harmony,
/// each with steps, confidence, and complexity assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fix {
    /// Description of the fix.
    pub description: String,
    /// Steps to implement the fix.
    pub steps: Vec<String>,
    /// Confidence that this fix will resolve the issue (0.0 to 1.0).
    pub confidence: f64,
    /// Complexity of implementing the fix.
    pub complexity: Complexity,
}

impl Fix {
    /// Create a new fix.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let fix = Fix::new("Refresh the API key", 0.9)
    ///     .with_steps(vec![
    ///         "Generate new API key from dashboard".to_string(),
    ///         "Update GEMINI_API_KEY environment variable".to_string(),
    ///         "Restart the application".to_string(),
    ///     ])
    ///     .with_complexity(Complexity::Simple);
    /// ```
    pub fn new(description: impl Into<String>, confidence: f64) -> Self {
        Self {
            description: description.into(),
            steps: Vec::new(),
            confidence: confidence.clamp(0.0, 1.0),
            complexity: Complexity::default(),
        }
    }

    /// Add implementation steps.
    pub fn with_steps(mut self, steps: Vec<String>) -> Self {
        self.steps = steps;
        self
    }

    /// Set the complexity.
    pub fn with_complexity(mut self, complexity: Complexity) -> Self {
        self.complexity = complexity;
        self
    }

    /// Add a single step.
    pub fn add_step(&mut self, step: impl Into<String>) {
        self.steps.push(step.into());
    }

    /// Check if this is a high-confidence fix.
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Check if this fix has implementation steps.
    pub fn has_steps(&self) -> bool {
        !self.steps.is_empty()
    }

    /// Get the estimated time to implement.
    pub fn estimated_time(&self) -> &'static str {
        self.complexity.estimated_time()
    }

    /// Format the fix as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!(
            "### {}\n\n**Confidence:** {:.0}% | **Complexity:** {} | **Time:** {}\n",
            self.description,
            self.confidence * 100.0,
            self.complexity,
            self.estimated_time()
        );

        if !self.steps.is_empty() {
            output.push_str("\n**Steps:**\n\n");
            for (i, step) in self.steps.iter().enumerate() {
                output.push_str(&format!("{}. {}\n", i + 1, step));
            }
        }

        output
    }
}

// =============================================================================
// DEBUG REPORT
// =============================================================================

/// A complete debug report for an issue.
///
/// The report contains the analyzed issue, identified root cause,
/// suggested fixes, and prevention tips for the future.
#[derive(Debug, Clone)]
pub struct DebugReport {
    /// The issue that was analyzed.
    pub issue: Issue,
    /// The identified root cause.
    pub root_cause: RootCause,
    /// Suggested fixes, ordered by confidence.
    pub suggested_fixes: Vec<Fix>,
    /// Tips to prevent this issue in the future.
    pub prevention_tips: Vec<String>,
    /// Processing duration in milliseconds.
    pub duration_ms: u64,
}

impl DebugReport {
    /// Create a new debug report.
    pub fn new(issue: Issue, root_cause: RootCause) -> Self {
        Self {
            issue,
            root_cause,
            suggested_fixes: Vec::new(),
            prevention_tips: Vec::new(),
            duration_ms: 0,
        }
    }

    /// Add suggested fixes.
    pub fn with_fixes(mut self, fixes: Vec<Fix>) -> Self {
        self.suggested_fixes = fixes;
        self
    }

    /// Add prevention tips.
    pub fn with_prevention_tips(mut self, tips: Vec<String>) -> Self {
        self.prevention_tips = tips;
        self
    }

    /// Get the best fix (highest confidence).
    pub fn best_fix(&self) -> Option<&Fix> {
        self.suggested_fixes
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }

    /// Get simple fixes only.
    pub fn simple_fixes(&self) -> Vec<&Fix> {
        self.suggested_fixes
            .iter()
            .filter(|f| f.complexity == Complexity::Simple)
            .collect()
    }

    /// Get high-confidence fixes only.
    pub fn high_confidence_fixes(&self) -> Vec<&Fix> {
        self.suggested_fixes
            .iter()
            .filter(|f| f.is_high_confidence())
            .collect()
    }

    /// Check if the report has actionable fixes.
    pub fn has_actionable_fixes(&self) -> bool {
        !self.suggested_fixes.is_empty()
    }

    /// Get a summary of the report.
    pub fn summary(&self) -> String {
        let fix_count = self.suggested_fixes.len();
        let best_confidence = self
            .best_fix()
            .map(|f| format!("{:.0}%", f.confidence * 100.0))
            .unwrap_or_else(|| "N/A".to_string());

        format!(
            "Issue: {} | Root cause confidence: {:.0}% | Fixes: {} (best: {})",
            self.issue.category,
            self.root_cause.confidence * 100.0,
            fix_count,
            best_confidence
        )
    }

    /// Format the report as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        // Issue section
        output.push_str("# Debug Report\n\n");
        output.push_str(&format!("## Issue Summary\n\n{}\n\n", self.issue.summary()));

        if let Some(ref context) = self.issue.context {
            output.push_str(&format!("**Context:** {}\n\n", context));
        }

        // Root cause section
        output.push_str(&self.root_cause.to_markdown());
        output.push('\n');

        // Fixes section
        if !self.suggested_fixes.is_empty() {
            output.push_str("## Suggested Fixes\n\n");
            for fix in &self.suggested_fixes {
                output.push_str(&fix.to_markdown());
                output.push('\n');
            }
        }

        // Prevention section
        if !self.prevention_tips.is_empty() {
            output.push_str("## Prevention Tips\n\n");
            for tip in &self.prevention_tips {
                output.push_str(&format!("- {}\n", tip));
            }
        }

        output
    }
}

// =============================================================================
// DEBUGGER AGENT CONFIGURATION
// =============================================================================

/// Configuration for the Debugger Agent.
#[derive(Debug, Clone)]
pub struct DebuggerConfig {
    /// Maximum number of fixes to suggest.
    pub max_fixes: usize,
    /// Maximum number of prevention tips to include.
    pub max_prevention_tips: usize,
    /// Minimum confidence threshold for including fixes.
    pub confidence_threshold: f64,
    /// Whether to include detailed evidence.
    pub include_evidence: bool,
    /// Timeout in seconds for analysis.
    pub timeout_secs: u64,
}

impl Default for DebuggerConfig {
    fn default() -> Self {
        Self {
            max_fixes: 5,
            max_prevention_tips: 3,
            confidence_threshold: 0.3,
            include_evidence: true,
            timeout_secs: 30,
        }
    }
}

impl DebuggerConfig {
    /// Create a fast configuration with fewer suggestions.
    pub fn fast() -> Self {
        Self {
            max_fixes: 2,
            max_prevention_tips: 1,
            confidence_threshold: 0.5,
            include_evidence: false,
            timeout_secs: 15,
        }
    }

    /// Create a thorough configuration with more analysis.
    pub fn thorough() -> Self {
        Self {
            max_fixes: 10,
            max_prevention_tips: 5,
            confidence_threshold: 0.2,
            include_evidence: true,
            timeout_secs: 60,
        }
    }
}

// =============================================================================
// DEBUGGER AGENT
// =============================================================================

/// The Error Hunter — analyzes issues and suggests fixes.
///
/// The Debugger Agent is a specialized component in the Sorcerer's Wand system
/// that traces problems back to their root causes and illuminates the path to
/// resolution. Like a master detective in the arcane arts, it combines logical
/// analysis with pattern recognition to break curses and restore harmony.
///
/// ## Capabilities
///
/// - **Issue Analysis**: Categorizes and assesses issue severity
/// - **Root Cause Detection**: Traces effects back to causes
/// - **Fix Suggestion**: Proposes solutions with confidence scores
/// - **Prevention Tips**: Provides guidance to avoid future issues
///
/// ## Example
///
/// ```rust,ignore
/// use panpsychism::debugger::{DebuggerAgent, Issue, IssueCategory, Severity};
///
/// let agent = DebuggerAgent::new();
///
/// let issue = Issue::new(
///     IssueCategory::TimeoutExceeded,
///     Severity::High,
///     "API request timed out after 30 seconds",
/// );
///
/// let report = agent.debug(&issue).await?;
/// println!("{}", report.to_markdown());
/// ```
#[derive(Debug, Clone)]
pub struct DebuggerAgent {
    /// Configuration for debugging behavior.
    config: DebuggerConfig,
}

impl Default for DebuggerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl DebuggerAgent {
    /// Create a new Debugger Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: DebuggerConfig::default(),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> DebuggerAgentBuilder {
        DebuggerAgentBuilder::default()
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: DebuggerConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &DebuggerConfig {
        &self.config
    }

    // =========================================================================
    // MAIN DEBUG METHOD
    // =========================================================================

    /// Analyze an issue and produce a debug report.
    ///
    /// This is the primary method of the Error Hunter, channeling analytical
    /// energy to trace issues back to their root causes and suggest fixes.
    ///
    /// # Arguments
    ///
    /// * `issue` - The issue to analyze
    ///
    /// # Returns
    ///
    /// A `DebugReport` containing the analysis and suggestions.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if analysis fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = DebuggerAgent::new();
    /// let issue = Issue::new(
    ///     IssueCategory::NetworkError,
    ///     Severity::Critical,
    ///     "Connection refused to API endpoint",
    /// );
    /// let report = agent.debug(&issue).await?;
    /// ```
    pub async fn debug(&self, issue: &Issue) -> Result<DebugReport> {
        let start = Instant::now();

        debug!(
            "Analyzing issue: {} ({}) - {}",
            issue.category, issue.severity, issue.description
        );

        // Analyze root cause
        let root_cause = self.analyze_root_cause(issue)?;

        // Generate fixes
        let suggested_fixes = self.generate_fixes(issue, &root_cause)?;

        // Generate prevention tips
        let prevention_tips = self.generate_prevention_tips(issue)?;

        let mut report = DebugReport::new(issue.clone(), root_cause)
            .with_fixes(suggested_fixes)
            .with_prevention_tips(prevention_tips);

        report.duration_ms = start.elapsed().as_millis() as u64;

        info!(
            "Debug analysis complete: {} fixes suggested in {}ms",
            report.suggested_fixes.len(),
            report.duration_ms
        );

        Ok(report)
    }

    // =========================================================================
    // ROOT CAUSE ANALYSIS
    // =========================================================================

    /// Analyze the root cause of an issue.
    fn analyze_root_cause(&self, issue: &Issue) -> Result<RootCause> {
        let typical_causes = issue.category.typical_causes();

        // Select the most likely cause based on description matching
        let (best_cause, confidence) = self.match_cause(issue, &typical_causes);

        let mut evidence = Vec::new();

        // Add evidence based on issue metadata
        if self.config.include_evidence {
            evidence.push(format!("Issue category: {}", issue.category));
            evidence.push(format!("Severity level: {}", issue.severity));

            if let Some(ref location) = issue.location {
                evidence.push(format!("Location: {}", location));
            }

            if let Some(ref context) = issue.context {
                evidence.push(format!("Context: {}", context));
            }

            // Add description-derived evidence
            if issue.description.contains("timeout") || issue.description.contains("slow") {
                evidence.push("Description mentions timing issues".to_string());
            }
            if issue.description.contains("error") || issue.description.contains("failed") {
                evidence.push("Description indicates failure condition".to_string());
            }
            if issue.description.contains("not found") || issue.description.contains("missing") {
                evidence.push("Description indicates missing resource".to_string());
            }
        }

        Ok(RootCause::new(best_cause, confidence).with_evidence(evidence))
    }

    /// Match the issue against typical causes and return the best match with confidence.
    fn match_cause(&self, issue: &Issue, typical_causes: &[&str]) -> (String, f64) {
        let description_lower = issue.description.to_lowercase();

        let mut best_match: Option<(&str, f64)> = None;

        for cause in typical_causes {
            let cause_lower = cause.to_lowercase();
            let mut score = 0.5; // Base score

            // Check for keyword matches
            let keywords: Vec<&str> = cause_lower.split_whitespace().collect();
            let matched_keywords = keywords
                .iter()
                .filter(|k| description_lower.contains(*k))
                .count();

            if matched_keywords > 0 {
                score += 0.1 * matched_keywords as f64;
            }

            // Check for exact phrase matches
            if description_lower.contains(&cause_lower) {
                score += 0.3;
            }

            // Context matching
            if let Some(ref context) = issue.context {
                let context_lower = context.to_lowercase();
                if keywords.iter().any(|k| context_lower.contains(*k)) {
                    score += 0.1;
                }
            }

            score = score.min(1.0);

            if best_match.is_none() || score > best_match.unwrap().1 {
                best_match = Some((cause, score));
            }
        }

        best_match
            .map(|(cause, score)| (cause.to_string(), score))
            .unwrap_or_else(|| {
                (
                    format!("Unknown cause for {} issue", issue.category),
                    0.3,
                )
            })
    }

    // =========================================================================
    // FIX GENERATION
    // =========================================================================

    /// Generate suggested fixes for an issue.
    fn generate_fixes(&self, issue: &Issue, _root_cause: &RootCause) -> Result<Vec<Fix>> {
        let fixes = match issue.category {
            IssueCategory::QueryMalformed => self.fixes_for_query_malformed(issue),
            IssueCategory::PromptNotFound => self.fixes_for_prompt_not_found(issue),
            IssueCategory::ValidationFailed => self.fixes_for_validation_failed(issue),
            IssueCategory::SynthesisFailed => self.fixes_for_synthesis_failed(issue),
            IssueCategory::TimeoutExceeded => self.fixes_for_timeout_exceeded(issue),
            IssueCategory::ResourceExhausted => self.fixes_for_resource_exhausted(issue),
            IssueCategory::ConfigurationError => self.fixes_for_configuration_error(issue),
            IssueCategory::NetworkError => self.fixes_for_network_error(issue),
        };

        // Filter by confidence threshold and limit
        let mut filtered: Vec<Fix> = fixes
            .into_iter()
            .filter(|f| f.confidence >= self.config.confidence_threshold)
            .take(self.config.max_fixes)
            .collect();

        // Sort by confidence (highest first)
        filtered.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(filtered)
    }

    /// Generate fixes for query malformed issues.
    fn fixes_for_query_malformed(&self, _issue: &Issue) -> Vec<Fix> {
        vec![
            Fix::new("Validate query before processing", 0.9)
                .with_steps(vec![
                    "Check that query is not empty".to_string(),
                    "Trim whitespace from query".to_string(),
                    "Validate query length is within limits".to_string(),
                    "Check for invalid characters".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Use query sanitization", 0.8)
                .with_steps(vec![
                    "Apply input sanitization to query".to_string(),
                    "Escape special characters".to_string(),
                    "Normalize Unicode characters".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Provide better error messages", 0.6)
                .with_steps(vec![
                    "Detect specific malformation type".to_string(),
                    "Generate user-friendly error message".to_string(),
                    "Suggest corrections to the user".to_string(),
                ])
                .with_complexity(Complexity::Moderate),
        ]
    }

    /// Generate fixes for prompt not found issues.
    fn fixes_for_prompt_not_found(&self, _issue: &Issue) -> Vec<Fix> {
        vec![
            Fix::new("Rebuild the prompt index", 0.85)
                .with_steps(vec![
                    "Run `panpsychism index --force`".to_string(),
                    "Verify prompts directory exists".to_string(),
                    "Check for file permission issues".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Check prompt ID format", 0.75)
                .with_steps(vec![
                    "Verify the prompt ID is valid".to_string(),
                    "Check for typos in the prompt name".to_string(),
                    "List available prompts with `panpsychism list`".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Search for similar prompts", 0.6)
                .with_steps(vec![
                    "Use fuzzy search to find similar prompts".to_string(),
                    "Suggest alternatives to the user".to_string(),
                    "Provide option to create new prompt".to_string(),
                ])
                .with_complexity(Complexity::Moderate),
        ]
    }

    /// Generate fixes for validation failed issues.
    fn fixes_for_validation_failed(&self, issue: &Issue) -> Vec<Fix> {
        let description_lower = issue.description.to_lowercase();
        let mut fixes = Vec::new();

        // CONATUS-specific fixes
        if description_lower.contains("conatus") {
            fixes.push(
                Fix::new("Clarify the purpose of your content", 0.85)
                    .with_steps(vec![
                        "Add a clear goal statement".to_string(),
                        "Specify the intended outcome".to_string(),
                        "Include action verbs".to_string(),
                    ])
                    .with_complexity(Complexity::Simple),
            );
        }

        // RATIO-specific fixes
        if description_lower.contains("ratio") {
            fixes.push(
                Fix::new("Improve logical structure", 0.85)
                    .with_steps(vec![
                        "Add clear headings and sections".to_string(),
                        "Use numbered steps for processes".to_string(),
                        "Ensure logical flow between paragraphs".to_string(),
                    ])
                    .with_complexity(Complexity::Moderate),
            );
        }

        // General validation fixes
        fixes.push(
            Fix::new("Revise content for clarity", 0.7)
                .with_steps(vec![
                    "Remove ambiguous language".to_string(),
                    "Use specific, concrete terms".to_string(),
                    "Break long sentences into shorter ones".to_string(),
                ])
                .with_complexity(Complexity::Moderate),
        );

        fixes.push(
            Fix::new("Lower validation threshold temporarily", 0.5)
                .with_steps(vec![
                    "Adjust the validation threshold in config".to_string(),
                    "Set a lower threshold for initial drafts".to_string(),
                    "Gradually increase as content improves".to_string(),
                ])
                .with_complexity(Complexity::Simple),
        );

        fixes
    }

    /// Generate fixes for synthesis failed issues.
    fn fixes_for_synthesis_failed(&self, _issue: &Issue) -> Vec<Fix> {
        vec![
            Fix::new("Check template variables", 0.85)
                .with_steps(vec![
                    "Verify all required variables are provided".to_string(),
                    "Check variable names match template".to_string(),
                    "Ensure variable values are valid".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Reduce context size", 0.75)
                .with_steps(vec![
                    "Summarize long context passages".to_string(),
                    "Remove redundant information".to_string(),
                    "Split into multiple smaller requests".to_string(),
                ])
                .with_complexity(Complexity::Moderate),
            Fix::new("Use fallback prompts", 0.6)
                .with_steps(vec![
                    "Configure fallback prompt templates".to_string(),
                    "Implement retry with simpler template".to_string(),
                    "Log failed synthesis for analysis".to_string(),
                ])
                .with_complexity(Complexity::Complex),
        ]
    }

    /// Generate fixes for timeout exceeded issues.
    fn fixes_for_timeout_exceeded(&self, _issue: &Issue) -> Vec<Fix> {
        vec![
            Fix::new("Increase timeout duration", 0.9)
                .with_steps(vec![
                    "Increase timeout setting in config".to_string(),
                    "Set --timeout flag for this request".to_string(),
                    "Consider if the operation should be async".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Simplify the request", 0.8)
                .with_steps(vec![
                    "Reduce input size".to_string(),
                    "Request fewer results".to_string(),
                    "Disable optional features".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Implement request caching", 0.6)
                .with_steps(vec![
                    "Enable response caching".to_string(),
                    "Configure cache TTL".to_string(),
                    "Use cached results when available".to_string(),
                ])
                .with_complexity(Complexity::Moderate),
            Fix::new("Use streaming responses", 0.5)
                .with_steps(vec![
                    "Enable streaming mode for LLM requests".to_string(),
                    "Process partial results as they arrive".to_string(),
                    "Show progress to user".to_string(),
                ])
                .with_complexity(Complexity::Complex),
        ]
    }

    /// Generate fixes for resource exhausted issues.
    fn fixes_for_resource_exhausted(&self, issue: &Issue) -> Vec<Fix> {
        let description_lower = issue.description.to_lowercase();
        let mut fixes = Vec::new();

        // Rate limit specific
        if description_lower.contains("rate limit") {
            fixes.push(
                Fix::new("Wait and retry", 0.95)
                    .with_steps(vec![
                        "Wait for rate limit window to reset".to_string(),
                        "Implement exponential backoff".to_string(),
                        "Queue requests for later processing".to_string(),
                    ])
                    .with_complexity(Complexity::Simple),
            );
            fixes.push(
                Fix::new("Reduce request frequency", 0.8)
                    .with_steps(vec![
                        "Add delays between requests".to_string(),
                        "Batch multiple operations".to_string(),
                        "Use bulk APIs where available".to_string(),
                    ])
                    .with_complexity(Complexity::Moderate),
            );
        }

        // Memory specific
        if description_lower.contains("memory") {
            fixes.push(
                Fix::new("Reduce memory usage", 0.85)
                    .with_steps(vec![
                        "Process data in smaller chunks".to_string(),
                        "Clear caches and buffers".to_string(),
                        "Use streaming instead of loading all data".to_string(),
                    ])
                    .with_complexity(Complexity::Moderate),
            );
        }

        // General resource fixes
        fixes.push(
            Fix::new("Scale resources", 0.5)
                .with_steps(vec![
                    "Increase memory allocation".to_string(),
                    "Add more processing capacity".to_string(),
                    "Consider load balancing".to_string(),
                ])
                .with_complexity(Complexity::Complex),
        );

        fixes
    }

    /// Generate fixes for configuration error issues.
    fn fixes_for_configuration_error(&self, _issue: &Issue) -> Vec<Fix> {
        vec![
            Fix::new("Check environment variables", 0.9)
                .with_steps(vec![
                    "Verify GEMINI_API_KEY is set".to_string(),
                    "Check other required environment variables".to_string(),
                    "Ensure values are not empty".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Validate configuration file", 0.85)
                .with_steps(vec![
                    "Run `panpsychism config --validate`".to_string(),
                    "Check for TOML syntax errors".to_string(),
                    "Verify all required fields are present".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Reset to default configuration", 0.7)
                .with_steps(vec![
                    "Backup current config".to_string(),
                    "Run `panpsychism init --force`".to_string(),
                    "Reconfigure with correct values".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Check file permissions", 0.5)
                .with_steps(vec![
                    "Verify read permissions on config file".to_string(),
                    "Check directory permissions".to_string(),
                    "Ensure user owns the config file".to_string(),
                ])
                .with_complexity(Complexity::Simple),
        ]
    }

    /// Generate fixes for network error issues.
    fn fixes_for_network_error(&self, _issue: &Issue) -> Vec<Fix> {
        vec![
            Fix::new("Check network connectivity", 0.9)
                .with_steps(vec![
                    "Verify internet connection is active".to_string(),
                    "Check if API endpoint is reachable".to_string(),
                    "Test with curl or ping".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Verify API endpoint configuration", 0.85)
                .with_steps(vec![
                    "Check the API endpoint URL is correct".to_string(),
                    "Verify Antigravity proxy is running".to_string(),
                    "Try alternative endpoint if available".to_string(),
                ])
                .with_complexity(Complexity::Simple),
            Fix::new("Check proxy and firewall settings", 0.7)
                .with_steps(vec![
                    "Verify proxy settings if used".to_string(),
                    "Check firewall rules".to_string(),
                    "Test direct connection vs proxy".to_string(),
                ])
                .with_complexity(Complexity::Moderate),
            Fix::new("Implement retry with backoff", 0.6)
                .with_steps(vec![
                    "Enable automatic retry on network errors".to_string(),
                    "Configure exponential backoff".to_string(),
                    "Set maximum retry attempts".to_string(),
                ])
                .with_complexity(Complexity::Moderate),
        ]
    }

    // =========================================================================
    // PREVENTION TIPS
    // =========================================================================

    /// Generate prevention tips for an issue.
    fn generate_prevention_tips(&self, issue: &Issue) -> Result<Vec<String>> {
        let tips = match issue.category {
            IssueCategory::QueryMalformed => vec![
                "Always validate user input before processing".to_string(),
                "Implement input length limits".to_string(),
                "Use input sanitization libraries".to_string(),
            ],
            IssueCategory::PromptNotFound => vec![
                "Rebuild the index after adding new prompts".to_string(),
                "Use prompt IDs from the list command".to_string(),
                "Set up file watchers for automatic reindexing".to_string(),
            ],
            IssueCategory::ValidationFailed => vec![
                "Review Spinoza principles before writing content".to_string(),
                "Use the corrector for iterative improvement".to_string(),
                "Start with clear, specific goals".to_string(),
            ],
            IssueCategory::SynthesisFailed => vec![
                "Test templates with sample data before use".to_string(),
                "Monitor context window usage".to_string(),
                "Keep fallback templates ready".to_string(),
            ],
            IssueCategory::TimeoutExceeded => vec![
                "Set appropriate timeout values for your use case".to_string(),
                "Consider using async operations for long tasks".to_string(),
                "Monitor API response times".to_string(),
            ],
            IssueCategory::ResourceExhausted => vec![
                "Monitor rate limit headers in responses".to_string(),
                "Implement request queuing".to_string(),
                "Cache frequently accessed data".to_string(),
            ],
            IssueCategory::ConfigurationError => vec![
                "Use environment variables for sensitive values".to_string(),
                "Validate configuration on startup".to_string(),
                "Document required configuration in README".to_string(),
            ],
            IssueCategory::NetworkError => vec![
                "Implement health checks for external services".to_string(),
                "Use circuit breakers for resilience".to_string(),
                "Have fallback endpoints configured".to_string(),
            ],
        };

        Ok(tips
            .into_iter()
            .take(self.config.max_prevention_tips)
            .collect())
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Create an issue from an error.
    pub fn issue_from_error(error: &Error) -> Issue {
        let (category, severity, description) = match error {
            Error::SearchEmptyQuery => (
                IssueCategory::QueryMalformed,
                Severity::Low,
                "Empty or invalid search query".to_string(),
            ),
            Error::SearchNoResults { query } => (
                IssueCategory::QueryMalformed,
                Severity::Low,
                format!("No results found for query: {}", query),
            ),
            Error::IndexNotFound { path } => (
                IssueCategory::PromptNotFound,
                Severity::Medium,
                format!("Index not found at {}", path),
            ),
            Error::ValidationFailed {
                principle,
                score,
                threshold,
            } => (
                IssueCategory::ValidationFailed,
                Severity::Medium,
                format!(
                    "{} validation failed: score {:.2} below threshold {:.2}",
                    principle, score, threshold
                ),
            ),
            Error::ValidationEmptyContent => (
                IssueCategory::ValidationFailed,
                Severity::Low,
                "Cannot validate empty content".to_string(),
            ),
            Error::SynthesisTemplateNotFound { template } => (
                IssueCategory::SynthesisFailed,
                Severity::Medium,
                format!("Synthesis template not found: {}", template),
            ),
            Error::SynthesisNoResponse => (
                IssueCategory::SynthesisFailed,
                Severity::High,
                "No response received from synthesis API".to_string(),
            ),
            Error::ApiTimeout { timeout_secs } => (
                IssueCategory::TimeoutExceeded,
                Severity::Medium,
                format!("API request timed out after {}s", timeout_secs),
            ),
            Error::ApiRateLimited { retry_after_secs } => {
                let desc = match retry_after_secs {
                    Some(secs) => format!("API rate limit exceeded, retry after {}s", secs),
                    None => "API rate limit exceeded".to_string(),
                };
                (IssueCategory::ResourceExhausted, Severity::Medium, desc)
            }
            Error::ConfigFileNotFound { path, .. } => (
                IssueCategory::ConfigurationError,
                Severity::High,
                format!("Configuration file not found: {}", path),
            ),
            Error::ConfigMissingKey { key } => (
                IssueCategory::ConfigurationError,
                Severity::High,
                format!("Missing required configuration: {}", key),
            ),
            Error::ApiConnectionFailed { endpoint, .. } => (
                IssueCategory::NetworkError,
                Severity::High,
                format!("Cannot connect to API at {}", endpoint),
            ),
            _ => (
                IssueCategory::ConfigurationError,
                Severity::Medium,
                error.to_string(),
            ),
        };

        Issue::new(category, severity, description)
    }

    /// Analyze an error directly and produce a debug report.
    pub async fn debug_error(&self, error: &Error) -> Result<DebugReport> {
        let issue = Self::issue_from_error(error);
        self.debug(&issue).await
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom DebuggerAgent configuration.
#[derive(Debug, Default)]
pub struct DebuggerAgentBuilder {
    config: Option<DebuggerConfig>,
}

impl DebuggerAgentBuilder {
    /// Set the configuration.
    pub fn config(mut self, config: DebuggerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set max fixes.
    pub fn max_fixes(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_fixes = max;
        self.config = Some(config);
        self
    }

    /// Set max prevention tips.
    pub fn max_prevention_tips(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_prevention_tips = max;
        self.config = Some(config);
        self
    }

    /// Set confidence threshold.
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.confidence_threshold = threshold.clamp(0.0, 1.0);
        self.config = Some(config);
        self
    }

    /// Set whether to include evidence.
    pub fn include_evidence(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_evidence = include;
        self.config = Some(config);
        self
    }

    /// Set timeout.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.timeout_secs = secs;
        self.config = Some(config);
        self
    }

    /// Build the DebuggerAgent.
    pub fn build(self) -> DebuggerAgent {
        DebuggerAgent {
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
    // IssueCategory Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_issue_category_display() {
        assert_eq!(IssueCategory::QueryMalformed.to_string(), "Query Malformed");
        assert_eq!(IssueCategory::PromptNotFound.to_string(), "Prompt Not Found");
        assert_eq!(
            IssueCategory::ValidationFailed.to_string(),
            "Validation Failed"
        );
        assert_eq!(
            IssueCategory::SynthesisFailed.to_string(),
            "Synthesis Failed"
        );
        assert_eq!(
            IssueCategory::TimeoutExceeded.to_string(),
            "Timeout Exceeded"
        );
        assert_eq!(
            IssueCategory::ResourceExhausted.to_string(),
            "Resource Exhausted"
        );
        assert_eq!(
            IssueCategory::ConfigurationError.to_string(),
            "Configuration Error"
        );
        assert_eq!(IssueCategory::NetworkError.to_string(), "Network Error");
    }

    #[test]
    fn test_issue_category_all() {
        let all = IssueCategory::all();
        assert_eq!(all.len(), 8);
        assert!(all.contains(&IssueCategory::QueryMalformed));
        assert!(all.contains(&IssueCategory::NetworkError));
    }

    #[test]
    fn test_issue_category_typical_causes() {
        let causes = IssueCategory::QueryMalformed.typical_causes();
        assert!(!causes.is_empty());
        assert!(causes.iter().any(|c| c.contains("Empty")));
    }

    #[test]
    fn test_issue_category_error_code() {
        assert_eq!(IssueCategory::QueryMalformed.error_code(), "E020");
        assert_eq!(IssueCategory::PromptNotFound.error_code(), "E010");
        assert_eq!(IssueCategory::ValidationFailed.error_code(), "E040");
        assert_eq!(IssueCategory::NetworkError.error_code(), "E030");
    }

    #[test]
    fn test_issue_category_from_str() {
        assert_eq!(
            "query_malformed".parse::<IssueCategory>().unwrap(),
            IssueCategory::QueryMalformed
        );
        assert_eq!(
            "malformed".parse::<IssueCategory>().unwrap(),
            IssueCategory::QueryMalformed
        );
        assert_eq!(
            "timeout".parse::<IssueCategory>().unwrap(),
            IssueCategory::TimeoutExceeded
        );
        assert_eq!(
            "network".parse::<IssueCategory>().unwrap(),
            IssueCategory::NetworkError
        );
    }

    #[test]
    fn test_issue_category_from_str_invalid() {
        let result = "invalid".parse::<IssueCategory>();
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Severity Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::Low.to_string(), "Low");
        assert_eq!(Severity::Medium.to_string(), "Medium");
        assert_eq!(Severity::High.to_string(), "High");
        assert_eq!(Severity::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_severity_all() {
        let all = Severity::all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], Severity::Low);
        assert_eq!(all[3], Severity::Critical);
    }

    #[test]
    fn test_severity_is_urgent() {
        assert!(!Severity::Low.is_urgent());
        assert!(!Severity::Medium.is_urgent());
        assert!(Severity::High.is_urgent());
        assert!(Severity::Critical.is_urgent());
    }

    #[test]
    fn test_severity_priority() {
        assert_eq!(Severity::Low.priority(), 1);
        assert_eq!(Severity::Medium.priority(), 2);
        assert_eq!(Severity::High.priority(), 3);
        assert_eq!(Severity::Critical.priority(), 4);
    }

    #[test]
    fn test_severity_response_time() {
        assert!(Severity::Low.response_time().contains("convenient"));
        assert!(Severity::Critical.response_time().contains("Immediately"));
    }

    #[test]
    fn test_severity_from_str() {
        assert_eq!("low".parse::<Severity>().unwrap(), Severity::Low);
        assert_eq!("medium".parse::<Severity>().unwrap(), Severity::Medium);
        assert_eq!("high".parse::<Severity>().unwrap(), Severity::High);
        assert_eq!("critical".parse::<Severity>().unwrap(), Severity::Critical);
        assert_eq!("l".parse::<Severity>().unwrap(), Severity::Low);
        assert_eq!("h".parse::<Severity>().unwrap(), Severity::High);
    }

    #[test]
    fn test_severity_from_str_invalid() {
        let result = "invalid".parse::<Severity>();
        assert!(result.is_err());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn test_severity_default() {
        assert_eq!(Severity::default(), Severity::Medium);
    }

    // -------------------------------------------------------------------------
    // Complexity Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_complexity_display() {
        assert_eq!(Complexity::Simple.to_string(), "Simple");
        assert_eq!(Complexity::Moderate.to_string(), "Moderate");
        assert_eq!(Complexity::Complex.to_string(), "Complex");
    }

    #[test]
    fn test_complexity_all() {
        let all = Complexity::all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_complexity_estimated_time() {
        assert!(Complexity::Simple.estimated_time().contains("30"));
        assert!(Complexity::Complex.estimated_time().contains("4"));
    }

    #[test]
    fn test_complexity_effort() {
        assert_eq!(Complexity::Simple.effort(), 1);
        assert_eq!(Complexity::Moderate.effort(), 2);
        assert_eq!(Complexity::Complex.effort(), 3);
    }

    #[test]
    fn test_complexity_from_str() {
        assert_eq!("simple".parse::<Complexity>().unwrap(), Complexity::Simple);
        assert_eq!("easy".parse::<Complexity>().unwrap(), Complexity::Simple);
        assert_eq!(
            "moderate".parse::<Complexity>().unwrap(),
            Complexity::Moderate
        );
        assert_eq!(
            "complex".parse::<Complexity>().unwrap(),
            Complexity::Complex
        );
    }

    #[test]
    fn test_complexity_from_str_invalid() {
        let result = "invalid".parse::<Complexity>();
        assert!(result.is_err());
    }

    #[test]
    fn test_complexity_default() {
        assert_eq!(Complexity::default(), Complexity::Moderate);
    }

    // -------------------------------------------------------------------------
    // Issue Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_issue_new() {
        let issue = Issue::new(
            IssueCategory::ValidationFailed,
            Severity::High,
            "Test issue",
        );
        assert_eq!(issue.category, IssueCategory::ValidationFailed);
        assert_eq!(issue.severity, Severity::High);
        assert_eq!(issue.description, "Test issue");
        assert!(issue.location.is_none());
        assert!(issue.context.is_none());
    }

    #[test]
    fn test_issue_builder() {
        let issue = Issue::new(IssueCategory::NetworkError, Severity::Critical, "Connection failed")
            .with_location("api::client")
            .with_context("Attempting to connect to Gemini API");

        assert_eq!(issue.location, Some("api::client".to_string()));
        assert!(issue.context.is_some());
    }

    #[test]
    fn test_issue_is_urgent() {
        let low = Issue::new(IssueCategory::QueryMalformed, Severity::Low, "test");
        assert!(!low.is_urgent());

        let critical = Issue::new(IssueCategory::NetworkError, Severity::Critical, "test");
        assert!(critical.is_urgent());
    }

    #[test]
    fn test_issue_error_code() {
        let issue = Issue::new(IssueCategory::ValidationFailed, Severity::High, "test");
        assert_eq!(issue.error_code(), "E040");
    }

    #[test]
    fn test_issue_summary() {
        let issue = Issue::new(
            IssueCategory::TimeoutExceeded,
            Severity::Medium,
            "API timed out",
        )
        .with_location("main.rs:42");

        let summary = issue.summary();
        assert!(summary.contains("E031"));
        assert!(summary.contains("Timeout Exceeded"));
        assert!(summary.contains("Medium"));
        assert!(summary.contains("main.rs:42"));
    }

    // -------------------------------------------------------------------------
    // RootCause Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_root_cause_new() {
        let cause = RootCause::new("API key expired", 0.95);
        assert_eq!(cause.description, "API key expired");
        assert!((cause.confidence - 0.95).abs() < f64::EPSILON);
        assert!(cause.evidence.is_empty());
    }

    #[test]
    fn test_root_cause_with_evidence() {
        let cause = RootCause::new("Connection refused", 0.8)
            .with_evidence(vec!["Port 443 not responding".to_string()]);

        assert_eq!(cause.evidence.len(), 1);
        assert!(cause.has_evidence());
    }

    #[test]
    fn test_root_cause_add_evidence() {
        let mut cause = RootCause::new("Test cause", 0.7);
        cause.add_evidence("Evidence 1");
        cause.add_evidence("Evidence 2");

        assert_eq!(cause.evidence.len(), 2);
    }

    #[test]
    fn test_root_cause_is_high_confidence() {
        let high = RootCause::new("High confidence cause", 0.9);
        assert!(high.is_high_confidence());

        let low = RootCause::new("Low confidence cause", 0.5);
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_root_cause_confidence_clamping() {
        let over = RootCause::new("Over", 1.5);
        assert!((over.confidence - 1.0).abs() < f64::EPSILON);

        let under = RootCause::new("Under", -0.5);
        assert!((under.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_root_cause_to_markdown() {
        let cause = RootCause::new("Test cause", 0.85)
            .with_evidence(vec!["Evidence A".to_string(), "Evidence B".to_string()]);

        let md = cause.to_markdown();
        assert!(md.contains("## Root Cause Analysis"));
        assert!(md.contains("Test cause"));
        assert!(md.contains("85%"));
        assert!(md.contains("Evidence A"));
    }

    // -------------------------------------------------------------------------
    // Fix Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fix_new() {
        let fix = Fix::new("Restart the service", 0.9);
        assert_eq!(fix.description, "Restart the service");
        assert!((fix.confidence - 0.9).abs() < f64::EPSILON);
        assert!(fix.steps.is_empty());
        assert_eq!(fix.complexity, Complexity::Moderate);
    }

    #[test]
    fn test_fix_builder() {
        let fix = Fix::new("Update configuration", 0.85)
            .with_steps(vec![
                "Edit config file".to_string(),
                "Restart service".to_string(),
            ])
            .with_complexity(Complexity::Simple);

        assert_eq!(fix.steps.len(), 2);
        assert_eq!(fix.complexity, Complexity::Simple);
    }

    #[test]
    fn test_fix_add_step() {
        let mut fix = Fix::new("Test fix", 0.8);
        fix.add_step("Step 1");
        fix.add_step("Step 2");

        assert_eq!(fix.steps.len(), 2);
        assert!(fix.has_steps());
    }

    #[test]
    fn test_fix_is_high_confidence() {
        let high = Fix::new("High", 0.85);
        assert!(high.is_high_confidence());

        let low = Fix::new("Low", 0.5);
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_fix_estimated_time() {
        let simple = Fix::new("Simple", 0.9).with_complexity(Complexity::Simple);
        assert!(simple.estimated_time().contains("30"));
    }

    #[test]
    fn test_fix_confidence_clamping() {
        let over = Fix::new("Over", 1.5);
        assert!((over.confidence - 1.0).abs() < f64::EPSILON);

        let under = Fix::new("Under", -0.5);
        assert!((under.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fix_to_markdown() {
        let fix = Fix::new("Test fix", 0.8)
            .with_steps(vec!["Step 1".to_string(), "Step 2".to_string()])
            .with_complexity(Complexity::Moderate);

        let md = fix.to_markdown();
        assert!(md.contains("### Test fix"));
        assert!(md.contains("80%"));
        assert!(md.contains("Moderate"));
        assert!(md.contains("1. Step 1"));
        assert!(md.contains("2. Step 2"));
    }

    // -------------------------------------------------------------------------
    // DebugReport Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_debug_report_new() {
        let issue = Issue::new(IssueCategory::NetworkError, Severity::High, "Connection failed");
        let cause = RootCause::new("Server is down", 0.9);
        let report = DebugReport::new(issue, cause);

        assert_eq!(report.issue.category, IssueCategory::NetworkError);
        assert_eq!(report.root_cause.description, "Server is down");
        assert!(report.suggested_fixes.is_empty());
        assert!(report.prevention_tips.is_empty());
    }

    #[test]
    fn test_debug_report_with_fixes() {
        let issue = Issue::new(IssueCategory::TimeoutExceeded, Severity::Medium, "Timeout");
        let cause = RootCause::new("Slow server", 0.7);
        let fixes = vec![
            Fix::new("Increase timeout", 0.9),
            Fix::new("Simplify request", 0.7),
        ];

        let report = DebugReport::new(issue, cause).with_fixes(fixes);
        assert_eq!(report.suggested_fixes.len(), 2);
        assert!(report.has_actionable_fixes());
    }

    #[test]
    fn test_debug_report_best_fix() {
        let issue = Issue::new(IssueCategory::QueryMalformed, Severity::Low, "Bad query");
        let cause = RootCause::new("Empty query", 0.95);
        let fixes = vec![
            Fix::new("Validate input", 0.9),
            Fix::new("Show error", 0.6),
        ];

        let report = DebugReport::new(issue, cause).with_fixes(fixes);
        let best = report.best_fix().unwrap();
        assert_eq!(best.description, "Validate input");
    }

    #[test]
    fn test_debug_report_simple_fixes() {
        let issue = Issue::new(IssueCategory::ConfigurationError, Severity::High, "Bad config");
        let cause = RootCause::new("Missing key", 0.9);
        let fixes = vec![
            Fix::new("Add key", 0.9).with_complexity(Complexity::Simple),
            Fix::new("Refactor", 0.7).with_complexity(Complexity::Complex),
        ];

        let report = DebugReport::new(issue, cause).with_fixes(fixes);
        let simple = report.simple_fixes();
        assert_eq!(simple.len(), 1);
        assert_eq!(simple[0].description, "Add key");
    }

    #[test]
    fn test_debug_report_high_confidence_fixes() {
        let issue = Issue::new(IssueCategory::SynthesisFailed, Severity::Medium, "Failed");
        let cause = RootCause::new("Template error", 0.8);
        let fixes = vec![Fix::new("High", 0.9), Fix::new("Low", 0.5)];

        let report = DebugReport::new(issue, cause).with_fixes(fixes);
        let high = report.high_confidence_fixes();
        assert_eq!(high.len(), 1);
    }

    #[test]
    fn test_debug_report_summary() {
        let issue = Issue::new(IssueCategory::ValidationFailed, Severity::Medium, "Failed");
        let cause = RootCause::new("Low clarity", 0.85);
        let fixes = vec![Fix::new("Improve", 0.8)];

        let report = DebugReport::new(issue, cause).with_fixes(fixes);
        let summary = report.summary();
        assert!(summary.contains("Validation Failed"));
        assert!(summary.contains("85%"));
        assert!(summary.contains("1"));
    }

    #[test]
    fn test_debug_report_to_markdown() {
        let issue = Issue::new(IssueCategory::NetworkError, Severity::Critical, "No connection")
            .with_context("Testing API");
        let cause = RootCause::new("Server offline", 0.9)
            .with_evidence(vec!["Connection refused".to_string()]);
        let fixes = vec![Fix::new("Check server", 0.95)
            .with_steps(vec!["Verify status".to_string()])
            .with_complexity(Complexity::Simple)];
        let tips = vec!["Monitor server health".to_string()];

        let report = DebugReport::new(issue, cause)
            .with_fixes(fixes)
            .with_prevention_tips(tips);

        let md = report.to_markdown();
        assert!(md.contains("# Debug Report"));
        assert!(md.contains("## Issue Summary"));
        assert!(md.contains("## Root Cause Analysis"));
        assert!(md.contains("## Suggested Fixes"));
        assert!(md.contains("## Prevention Tips"));
        assert!(md.contains("Monitor server health"));
    }

    // -------------------------------------------------------------------------
    // DebuggerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_debugger_config_default() {
        let config = DebuggerConfig::default();
        assert_eq!(config.max_fixes, 5);
        assert_eq!(config.max_prevention_tips, 3);
        assert!((config.confidence_threshold - 0.3).abs() < f64::EPSILON);
        assert!(config.include_evidence);
    }

    #[test]
    fn test_debugger_config_fast() {
        let config = DebuggerConfig::fast();
        assert_eq!(config.max_fixes, 2);
        assert_eq!(config.timeout_secs, 15);
        assert!(!config.include_evidence);
    }

    #[test]
    fn test_debugger_config_thorough() {
        let config = DebuggerConfig::thorough();
        assert_eq!(config.max_fixes, 10);
        assert_eq!(config.max_prevention_tips, 5);
    }

    // -------------------------------------------------------------------------
    // DebuggerAgent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_debugger_agent_new() {
        let agent = DebuggerAgent::new();
        assert_eq!(agent.config.max_fixes, 5);
    }

    #[test]
    fn test_debugger_agent_builder() {
        let agent = DebuggerAgent::builder()
            .max_fixes(10)
            .max_prevention_tips(4)
            .confidence_threshold(0.5)
            .include_evidence(false)
            .timeout_secs(60)
            .build();

        assert_eq!(agent.config.max_fixes, 10);
        assert_eq!(agent.config.max_prevention_tips, 4);
        assert!((agent.config.confidence_threshold - 0.5).abs() < f64::EPSILON);
        assert!(!agent.config.include_evidence);
        assert_eq!(agent.config.timeout_secs, 60);
    }

    #[test]
    fn test_debugger_agent_with_config() {
        let config = DebuggerConfig::fast();
        let agent = DebuggerAgent::new().with_config(config);
        assert_eq!(agent.config.max_fixes, 2);
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_query_malformed() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::QueryMalformed,
            Severity::Low,
            "Empty query provided",
        );

        let report = agent.debug(&issue).await.unwrap();
        assert_eq!(report.issue.category, IssueCategory::QueryMalformed);
        assert!(!report.suggested_fixes.is_empty());
        assert!(!report.prevention_tips.is_empty());
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_prompt_not_found() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::PromptNotFound,
            Severity::Medium,
            "Prompt file was deleted",
        );

        let report = agent.debug(&issue).await.unwrap();
        assert!(report.root_cause.confidence > 0.0);
        assert!(!report.suggested_fixes.is_empty());
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_validation_failed() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::ValidationFailed,
            Severity::Medium,
            "CONATUS score too low",
        );

        let report = agent.debug(&issue).await.unwrap();
        assert!(report
            .suggested_fixes
            .iter()
            .any(|f| f.description.to_lowercase().contains("purpose")
                || f.description.to_lowercase().contains("clarity")));
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_synthesis_failed() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::SynthesisFailed,
            Severity::High,
            "Template variable not found",
        );

        let report = agent.debug(&issue).await.unwrap();
        assert!(report
            .suggested_fixes
            .iter()
            .any(|f| f.description.to_lowercase().contains("template")));
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_timeout_exceeded() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::TimeoutExceeded,
            Severity::Medium,
            "API request timed out after 30s",
        );

        let report = agent.debug(&issue).await.unwrap();
        assert!(report
            .suggested_fixes
            .iter()
            .any(|f| f.description.to_lowercase().contains("timeout")));
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_resource_exhausted() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::ResourceExhausted,
            Severity::High,
            "Rate limit exceeded",
        );

        let report = agent.debug(&issue).await.unwrap();
        assert!(report
            .suggested_fixes
            .iter()
            .any(|f| f.description.to_lowercase().contains("wait")
                || f.description.to_lowercase().contains("retry")));
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_configuration_error() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::ConfigurationError,
            Severity::High,
            "API key missing",
        );

        let report = agent.debug(&issue).await.unwrap();
        assert!(report
            .suggested_fixes
            .iter()
            .any(|f| f.description.to_lowercase().contains("environment")
                || f.description.to_lowercase().contains("config")));
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_network_error() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::NetworkError,
            Severity::Critical,
            "Connection refused",
        );

        let report = agent.debug(&issue).await.unwrap();
        assert!(report
            .suggested_fixes
            .iter()
            .any(|f| f.description.to_lowercase().contains("network")
                || f.description.to_lowercase().contains("connectivity")));
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_with_context() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::ValidationFailed,
            Severity::Medium,
            "RATIO validation failed",
        )
        .with_location("validator.rs:123")
        .with_context("User submitted content with unclear structure");

        let report = agent.debug(&issue).await.unwrap();
        assert!(report.root_cause.has_evidence());
        assert!(report
            .root_cause
            .evidence
            .iter()
            .any(|e| e.contains("Location")));
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_report_duration() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(IssueCategory::QueryMalformed, Severity::Low, "Bad query");

        let report = agent.debug(&issue).await.unwrap();
        // Duration may be 0 on very fast systems, just ensure it doesn't panic
        // The important thing is that the field is populated
        assert!(report.duration_ms <= 10000); // Should complete within 10 seconds
    }

    #[tokio::test]
    async fn test_debugger_agent_confidence_threshold() {
        let agent = DebuggerAgent::builder()
            .confidence_threshold(0.8)
            .build();

        let issue = Issue::new(IssueCategory::QueryMalformed, Severity::Low, "Test");
        let report = agent.debug(&issue).await.unwrap();

        // All fixes should have confidence >= 0.8
        for fix in &report.suggested_fixes {
            assert!(fix.confidence >= 0.8);
        }
    }

    #[tokio::test]
    async fn test_debugger_agent_max_fixes() {
        let agent = DebuggerAgent::builder().max_fixes(2).build();

        let issue = Issue::new(
            IssueCategory::NetworkError,
            Severity::High,
            "Connection failed",
        );
        let report = agent.debug(&issue).await.unwrap();

        assert!(report.suggested_fixes.len() <= 2);
    }

    #[tokio::test]
    async fn test_debugger_agent_max_prevention_tips() {
        let agent = DebuggerAgent::builder().max_prevention_tips(1).build();

        let issue = Issue::new(
            IssueCategory::TimeoutExceeded,
            Severity::Medium,
            "Timeout",
        );
        let report = agent.debug(&issue).await.unwrap();

        assert!(report.prevention_tips.len() <= 1);
    }

    // -------------------------------------------------------------------------
    // Issue from Error Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_issue_from_error_search_empty() {
        let error = Error::SearchEmptyQuery;
        let issue = DebuggerAgent::issue_from_error(&error);

        assert_eq!(issue.category, IssueCategory::QueryMalformed);
        assert_eq!(issue.severity, Severity::Low);
    }

    #[test]
    fn test_issue_from_error_validation_failed() {
        let error = Error::ValidationFailed {
            principle: "CONATUS".to_string(),
            score: 0.3,
            threshold: 0.5,
        };
        let issue = DebuggerAgent::issue_from_error(&error);

        assert_eq!(issue.category, IssueCategory::ValidationFailed);
        assert!(issue.description.contains("CONATUS"));
    }

    #[test]
    fn test_issue_from_error_api_timeout() {
        let error = Error::ApiTimeout { timeout_secs: 30 };
        let issue = DebuggerAgent::issue_from_error(&error);

        assert_eq!(issue.category, IssueCategory::TimeoutExceeded);
        assert!(issue.description.contains("30"));
    }

    #[test]
    fn test_issue_from_error_rate_limit() {
        let error = Error::ApiRateLimited {
            retry_after_secs: Some(60),
        };
        let issue = DebuggerAgent::issue_from_error(&error);

        assert_eq!(issue.category, IssueCategory::ResourceExhausted);
        assert!(issue.description.contains("60"));
    }

    #[test]
    fn test_issue_from_error_config_not_found() {
        let error = Error::ConfigFileNotFound {
            path: "/path/to/config.toml".to_string(),
            source: None,
        };
        let issue = DebuggerAgent::issue_from_error(&error);

        assert_eq!(issue.category, IssueCategory::ConfigurationError);
        assert_eq!(issue.severity, Severity::High);
    }

    #[test]
    fn test_issue_from_error_network() {
        let error = Error::ApiConnectionFailed {
            endpoint: "http://localhost:8045".to_string(),
            source: None,
        };
        let issue = DebuggerAgent::issue_from_error(&error);

        assert_eq!(issue.category, IssueCategory::NetworkError);
        assert_eq!(issue.severity, Severity::High);
    }

    #[tokio::test]
    async fn test_debugger_agent_debug_error() {
        let agent = DebuggerAgent::new();
        let error = Error::ApiTimeout { timeout_secs: 30 };

        let report = agent.debug_error(&error).await.unwrap();
        assert_eq!(report.issue.category, IssueCategory::TimeoutExceeded);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_issue_category_causes_not_empty() {
        for category in IssueCategory::all() {
            assert!(!category.typical_causes().is_empty());
        }
    }

    #[test]
    fn test_severity_all_ordered() {
        let all = Severity::all();
        for i in 0..all.len() - 1 {
            assert!(all[i] < all[i + 1]);
        }
    }

    #[test]
    fn test_fix_steps_ordered_in_markdown() {
        let fix = Fix::new("Test", 0.9)
            .with_steps(vec!["First".to_string(), "Second".to_string(), "Third".to_string()]);

        let md = fix.to_markdown();
        let first_pos = md.find("1. First").unwrap();
        let second_pos = md.find("2. Second").unwrap();
        let third_pos = md.find("3. Third").unwrap();

        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }

    #[test]
    fn test_debug_report_no_fixes() {
        let issue = Issue::new(IssueCategory::QueryMalformed, Severity::Low, "Test");
        let cause = RootCause::new("Unknown", 0.3);
        let report = DebugReport::new(issue, cause);

        assert!(!report.has_actionable_fixes());
        assert!(report.best_fix().is_none());
    }

    #[tokio::test]
    async fn test_debugger_fixes_sorted_by_confidence() {
        let agent = DebuggerAgent::new();
        let issue = Issue::new(
            IssueCategory::NetworkError,
            Severity::High,
            "Connection failed",
        );

        let report = agent.debug(&issue).await.unwrap();

        // Verify fixes are sorted by confidence (descending)
        for i in 0..report.suggested_fixes.len().saturating_sub(1) {
            assert!(report.suggested_fixes[i].confidence >= report.suggested_fixes[i + 1].confidence);
        }
    }
}
