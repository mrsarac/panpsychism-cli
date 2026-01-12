//! # Auditor Agent Module for Project Panpsychism
//!
//! The Compliance Sentinel - "Every spell must pass through the gates of law."
//!
//! This module implements the Auditor Agent (Agent 26), responsible for auditing
//! prompts and responses for compliance with policies and regulations. Like a
//! vigilant sentinel at the gates of the magical realm, the Auditor ensures all
//! content adheres to established rules and standards.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, The Compliance Sentinel serves as the guardian:
//!
//! - **Content** (magical output) approaches the sentinel's gate
//! - **The Sentinel** (AuditorAgent) examines against the book of laws
//! - **Judgment** (AuditReport) reveals compliance status and findings
//!
//! The Sentinel audits for:
//! - **Policy Compliance**: Adherence to organizational policies
//! - **Regulatory Standards**: GDPR, HIPAA, ethical AI guidelines
//! - **Content Safety**: Harmful content detection
//! - **Format Standards**: Structural and formatting rules
//! - **Privacy Protection**: PII and sensitive data handling
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//!
//! - **CONATUS**: Self-preservation through compliance safeguards
//! - **RATIO**: Logical application of rules and policies
//! - **LAETITIA**: Joy through trust and safety
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::auditor::{AuditorAgent, AuditLevel, PolicySet};
//!
//! let sentinel = AuditorAgent::builder()
//!     .audit_level(AuditLevel::Comprehensive)
//!     .build();
//!
//! // Audit content against policies
//! let report = sentinel.audit(
//!     "Here is the response content...",
//!     Some("Original prompt"),
//! ).await?;
//!
//! println!("Compliance score: {:.2}", report.compliance_score);
//! for finding in &report.findings {
//!     println!("[{}] {}: {}", finding.severity, finding.rule_id, finding.description);
//! }
//! ```

use crate::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tracing::{debug, info, warn};

// =============================================================================
// AUDIT SEVERITY
// =============================================================================

/// Severity level for audit findings.
///
/// Like the urgency of magical infractions, severity indicates
/// how critical a compliance violation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum AuditSeverity {
    /// Informational observation, no action required.
    Informational,
    /// Minor issue that should be addressed but is not blocking.
    Minor,
    /// Major issue that requires attention before deployment.
    Major,
    /// Critical violation that must be fixed immediately.
    Critical,
}

impl Default for AuditSeverity {
    fn default() -> Self {
        Self::Minor
    }
}

impl std::fmt::Display for AuditSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Informational => write!(f, "INFO"),
            Self::Minor => write!(f, "MINOR"),
            Self::Major => write!(f, "MAJOR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl std::str::FromStr for AuditSeverity {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "informational" | "info" | "note" => Ok(Self::Informational),
            "minor" | "low" | "warning" => Ok(Self::Minor),
            "major" | "medium" | "high" => Ok(Self::Major),
            "critical" | "blocker" | "severe" => Ok(Self::Critical),
            _ => Err(Error::Config(format!(
                "Unknown audit severity: '{}'. Valid: informational, minor, major, critical",
                s
            ))),
        }
    }
}

impl AuditSeverity {
    /// Get all severity levels in order of increasing severity.
    pub fn all() -> Vec<Self> {
        vec![Self::Informational, Self::Minor, Self::Major, Self::Critical]
    }

    /// Get a numeric weight for scoring (0-3).
    pub fn weight(&self) -> u32 {
        match self {
            Self::Informational => 0,
            Self::Minor => 1,
            Self::Major => 2,
            Self::Critical => 3,
        }
    }

    /// Check if this severity is at least as severe as another.
    pub fn is_at_least(&self, other: Self) -> bool {
        self.weight() >= other.weight()
    }
}

// =============================================================================
// AUDIT LEVEL
// =============================================================================

/// Depth of audit to perform.
///
/// Like different levels of magical inspection, each level
/// provides varying degrees of thoroughness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AuditLevel {
    /// Quick audit with basic checks.
    /// Fast but less thorough - good for development.
    Quick,

    /// Standard audit with comprehensive analysis.
    /// Balanced between speed and thoroughness.
    #[default]
    Standard,

    /// Comprehensive audit with exhaustive analysis.
    /// Thorough but slower - required for production deployment.
    Comprehensive,
}

impl std::fmt::Display for AuditLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Quick => write!(f, "quick"),
            Self::Standard => write!(f, "standard"),
            Self::Comprehensive => write!(f, "comprehensive"),
        }
    }
}

impl std::str::FromStr for AuditLevel {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "quick" | "fast" | "basic" => Ok(Self::Quick),
            "standard" | "normal" | "default" => Ok(Self::Standard),
            "comprehensive" | "full" | "exhaustive" | "deep" => Ok(Self::Comprehensive),
            _ => Err(Error::Config(format!(
                "Unknown audit level: '{}'. Valid: quick, standard, comprehensive",
                s
            ))),
        }
    }
}

impl AuditLevel {
    /// Get all audit levels.
    pub fn all() -> Vec<Self> {
        vec![Self::Quick, Self::Standard, Self::Comprehensive]
    }

    /// Get the expected duration multiplier for this level.
    pub fn duration_multiplier(&self) -> f64 {
        match self {
            Self::Quick => 0.5,
            Self::Standard => 1.0,
            Self::Comprehensive => 2.5,
        }
    }

    /// Get the number of rule categories to check at this level.
    pub fn rule_depth(&self) -> usize {
        match self {
            Self::Quick => 2,      // Basic safety + format
            Self::Standard => 4,   // + privacy + policy
            Self::Comprehensive => 6, // + regulatory + custom
        }
    }
}

// =============================================================================
// POLICY RULE
// =============================================================================

/// A single policy rule for compliance checking.
///
/// Each rule defines a pattern to match against and the
/// resulting severity if matched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Unique identifier for this rule.
    pub id: String,
    /// Human-readable name of the rule.
    pub name: String,
    /// Regex pattern to match against content.
    pub pattern: String,
    /// Compiled regex (transient, not serialized).
    #[serde(skip)]
    compiled_pattern: Option<Regex>,
    /// Severity if this rule is violated.
    pub severity: AuditSeverity,
    /// Human-readable message describing the violation.
    pub message: String,
    /// Category this rule belongs to.
    pub category: String,
    /// Optional remediation guidance.
    pub remediation: Option<String>,
    /// Whether this rule is enabled.
    pub enabled: bool,
}

impl PolicyRule {
    /// Create a new policy rule.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        pattern: impl Into<String>,
        severity: AuditSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            pattern: pattern.into(),
            compiled_pattern: None,
            severity,
            message: message.into(),
            category: "general".to_string(),
            remediation: None,
            enabled: true,
        }
    }

    /// Set the category for this rule.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }

    /// Set the remediation guidance for this rule.
    pub fn with_remediation(mut self, remediation: impl Into<String>) -> Self {
        self.remediation = Some(remediation.into());
        self
    }

    /// Enable or disable this rule.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Compile the regex pattern.
    pub fn compile(&mut self) -> Result<()> {
        let regex = Regex::new(&self.pattern)
            .map_err(|e| Error::Config(format!("Invalid regex in rule {}: {}", self.id, e)))?;
        self.compiled_pattern = Some(regex);
        Ok(())
    }

    /// Check if the pattern matches the content.
    pub fn matches(&self, content: &str) -> bool {
        if let Some(ref regex) = self.compiled_pattern {
            regex.is_match(content)
        } else if let Ok(regex) = Regex::new(&self.pattern) {
            regex.is_match(content)
        } else {
            false
        }
    }

    /// Find all matches in the content.
    pub fn find_matches(&self, content: &str) -> Vec<(usize, usize)> {
        let regex = self.compiled_pattern.as_ref()
            .cloned()
            .or_else(|| Regex::new(&self.pattern).ok());

        if let Some(regex) = regex {
            regex.find_iter(content)
                .map(|m| (m.start(), m.end()))
                .collect()
        } else {
            Vec::new()
        }
    }
}

// =============================================================================
// POLICY SET
// =============================================================================

/// A collection of policy rules for a specific domain or purpose.
///
/// Policy sets can be combined and customized for different
/// compliance requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicySet {
    /// Unique identifier for this policy set.
    pub name: String,
    /// Description of what this policy set covers.
    pub description: String,
    /// Version of this policy set.
    pub version: String,
    /// Rules in this policy set.
    pub rules: Vec<PolicyRule>,
    /// Whether this policy set is enabled.
    pub enabled: bool,
}

impl PolicySet {
    /// Create a new policy set.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            version: version.into(),
            rules: Vec::new(),
            enabled: true,
        }
    }

    /// Add a rule to this policy set.
    pub fn add_rule(mut self, rule: PolicyRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add multiple rules to this policy set.
    pub fn add_rules(mut self, rules: Vec<PolicyRule>) -> Self {
        self.rules.extend(rules);
        self
    }

    /// Enable or disable this policy set.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Get enabled rules.
    pub fn enabled_rules(&self) -> Vec<&PolicyRule> {
        self.rules.iter().filter(|r| r.enabled).collect()
    }

    /// Get rules by category.
    pub fn rules_by_category(&self, category: &str) -> Vec<&PolicyRule> {
        self.rules.iter()
            .filter(|r| r.enabled && r.category == category)
            .collect()
    }

    /// Compile all rule patterns.
    pub fn compile_all(&mut self) -> Result<()> {
        for rule in &mut self.rules {
            rule.compile()?;
        }
        Ok(())
    }

    /// Get the number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Get the number of enabled rules.
    pub fn enabled_rule_count(&self) -> usize {
        self.rules.iter().filter(|r| r.enabled).count()
    }
}

impl Default for PolicySet {
    fn default() -> Self {
        Self::new("default", "Default policy set", "1.0.0")
    }
}

// =============================================================================
// AUDIT FINDING
// =============================================================================

/// A single compliance finding from an audit.
///
/// Like a note from the sentinel's inspection, each finding
/// documents a specific issue or observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditFinding {
    /// The rule ID that triggered this finding.
    pub rule_id: String,
    /// Severity of this finding.
    pub severity: AuditSeverity,
    /// Description of what was found.
    pub description: String,
    /// Location in the content (character offset, if applicable).
    pub location: Option<AuditLocation>,
    /// Suggested remediation for this finding.
    pub remediation: Option<String>,
    /// Category of the finding.
    pub category: String,
    /// Additional context or evidence.
    pub context: Option<String>,
}

impl AuditFinding {
    /// Create a new audit finding.
    pub fn new(
        rule_id: impl Into<String>,
        severity: AuditSeverity,
        description: impl Into<String>,
    ) -> Self {
        Self {
            rule_id: rule_id.into(),
            severity,
            description: description.into(),
            location: None,
            remediation: None,
            category: "general".to_string(),
            context: None,
        }
    }

    /// Set the location for this finding.
    pub fn with_location(mut self, start: usize, end: usize) -> Self {
        self.location = Some(AuditLocation { start, end, line: None, column: None });
        self
    }

    /// Set the full location including line and column.
    pub fn with_full_location(mut self, start: usize, end: usize, line: usize, column: usize) -> Self {
        self.location = Some(AuditLocation {
            start,
            end,
            line: Some(line),
            column: Some(column),
        });
        self
    }

    /// Set the remediation guidance.
    pub fn with_remediation(mut self, remediation: impl Into<String>) -> Self {
        self.remediation = Some(remediation.into());
        self
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }

    /// Set additional context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Format the finding as a single line for display.
    pub fn to_line(&self) -> String {
        let location_str = self.location.as_ref()
            .map(|l| format!(" at {}:{}", l.start, l.end))
            .unwrap_or_default();
        format!(
            "[{}] {}{}: {}",
            self.severity, self.rule_id, location_str, self.description
        )
    }

    /// Format the finding as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = format!(
            "- **[{}] {}**: {}\n",
            self.severity, self.rule_id, self.description
        );

        if let Some(ref location) = self.location {
            output.push_str(&format!("  - Location: {}-{}\n", location.start, location.end));
            if let (Some(line), Some(col)) = (location.line, location.column) {
                output.push_str(&format!("  - Line {}, Column {}\n", line, col));
            }
        }

        if let Some(ref remediation) = self.remediation {
            output.push_str(&format!("  - Remediation: {}\n", remediation));
        }

        if let Some(ref context) = self.context {
            output.push_str(&format!("  - Context: `{}`\n", context));
        }

        output
    }
}

/// Location information for an audit finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLocation {
    /// Start character offset.
    pub start: usize,
    /// End character offset.
    pub end: usize,
    /// Line number (1-based), if available.
    pub line: Option<usize>,
    /// Column number (1-based), if available.
    pub column: Option<usize>,
}

// =============================================================================
// AUDIT REPORT
// =============================================================================

/// Complete audit report from the Compliance Sentinel.
///
/// Contains the compliance score, all findings, and recommendations
/// for improving compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// Overall compliance score (0.0 - 1.0).
    /// 1.0 means fully compliant, 0.0 means completely non-compliant.
    pub compliance_score: f64,

    /// All findings from the audit.
    pub findings: Vec<AuditFinding>,

    /// High-level recommendations for improving compliance.
    pub recommendations: Vec<String>,

    /// The audit level that was used.
    pub level: AuditLevel,

    /// Policy sets that were checked.
    pub policies_checked: Vec<String>,

    /// Number of rules evaluated.
    pub rules_evaluated: u32,

    /// Processing duration in milliseconds.
    pub duration_ms: u64,

    /// Length of the audited content.
    pub content_length: usize,

    /// Whether the content passed the audit (no critical/major findings).
    pub passed: bool,

    /// Summary statistics by severity.
    pub severity_counts: HashMap<String, u32>,

    /// Summary statistics by category.
    pub category_counts: HashMap<String, u32>,
}

impl AuditReport {
    /// Create a new audit report.
    pub fn new(compliance_score: f64, level: AuditLevel) -> Self {
        Self {
            compliance_score: compliance_score.clamp(0.0, 1.0),
            findings: Vec::new(),
            recommendations: Vec::new(),
            level,
            policies_checked: Vec::new(),
            rules_evaluated: 0,
            duration_ms: 0,
            content_length: 0,
            passed: true,
            severity_counts: HashMap::new(),
            category_counts: HashMap::new(),
        }
    }

    /// Check if the audit passed (no critical or major findings).
    pub fn is_passed(&self) -> bool {
        !self.findings.iter().any(|f| f.severity.is_at_least(AuditSeverity::Major))
    }

    /// Check if there are any critical findings.
    pub fn has_critical(&self) -> bool {
        self.findings.iter().any(|f| f.severity == AuditSeverity::Critical)
    }

    /// Check if there are any major findings.
    pub fn has_major(&self) -> bool {
        self.findings.iter().any(|f| f.severity == AuditSeverity::Major)
    }

    /// Get findings by severity.
    pub fn findings_by_severity(&self, severity: AuditSeverity) -> Vec<&AuditFinding> {
        self.findings.iter().filter(|f| f.severity == severity).collect()
    }

    /// Get findings by category.
    pub fn findings_by_category(&self, category: &str) -> Vec<&AuditFinding> {
        self.findings.iter().filter(|f| f.category == category).collect()
    }

    /// Get the count of findings for a severity level.
    pub fn count_by_severity(&self, severity: AuditSeverity) -> usize {
        self.findings.iter().filter(|f| f.severity == severity).count()
    }

    /// Get a summary of the audit.
    pub fn summary(&self) -> String {
        let status = if self.passed { "PASSED" } else { "FAILED" };
        let critical_count = self.count_by_severity(AuditSeverity::Critical);
        let major_count = self.count_by_severity(AuditSeverity::Major);
        let minor_count = self.count_by_severity(AuditSeverity::Minor);

        format!(
            "{} ({:.0}%) - {} critical, {} major, {} minor findings",
            status,
            self.compliance_score * 100.0,
            critical_count,
            major_count,
            minor_count
        )
    }

    /// Format the report as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Audit Report\n\n");
        output.push_str(&format!(
            "**Status:** {} | **Compliance Score:** {:.0}%\n\n",
            if self.passed { "PASSED" } else { "FAILED" },
            self.compliance_score * 100.0
        ));
        output.push_str(&format!(
            "**Level:** {} | **Rules Evaluated:** {} | **Duration:** {}ms\n\n",
            self.level, self.rules_evaluated, self.duration_ms
        ));

        if !self.findings.is_empty() {
            output.push_str("## Findings\n\n");

            // Group by severity
            for severity in [AuditSeverity::Critical, AuditSeverity::Major, AuditSeverity::Minor, AuditSeverity::Informational] {
                let findings: Vec<_> = self.findings_by_severity(severity);
                if !findings.is_empty() {
                    output.push_str(&format!("### {} ({})\n\n", severity, findings.len()));
                    for finding in findings {
                        output.push_str(&finding.to_markdown());
                    }
                    output.push('\n');
                }
            }
        }

        if !self.recommendations.is_empty() {
            output.push_str("## Recommendations\n\n");
            for rec in &self.recommendations {
                output.push_str(&format!("- {}\n", rec));
            }
            output.push('\n');
        }

        if !self.policies_checked.is_empty() {
            output.push_str("## Policies Checked\n\n");
            for policy in &self.policies_checked {
                output.push_str(&format!("- {}\n", policy));
            }
        }

        output
    }
}

// =============================================================================
// AUDITOR CONFIGURATION
// =============================================================================

/// Configuration for the Auditor Agent.
#[derive(Debug, Clone)]
pub struct AuditorConfig {
    /// Default audit level.
    pub default_level: AuditLevel,

    /// Minimum severity to include in findings.
    pub min_severity: AuditSeverity,

    /// Maximum number of findings to report.
    pub max_findings: usize,

    /// Whether to fail on major violations.
    pub fail_on_major: bool,

    /// Whether to fail on critical violations.
    pub fail_on_critical: bool,

    /// Timeout for audit operations in seconds.
    pub timeout_secs: u64,

    /// Whether to include context snippets in findings.
    pub include_context: bool,

    /// Maximum context length to include.
    pub context_length: usize,

    /// Custom policy sets to use.
    pub custom_policies: Vec<PolicySet>,

    /// Categories to enable (empty = all).
    pub enabled_categories: HashSet<String>,

    /// Categories to disable.
    pub disabled_categories: HashSet<String>,
}

impl Default for AuditorConfig {
    fn default() -> Self {
        Self {
            default_level: AuditLevel::Standard,
            min_severity: AuditSeverity::Informational,
            max_findings: 100,
            fail_on_major: true,
            fail_on_critical: true,
            timeout_secs: 30,
            include_context: true,
            context_length: 50,
            custom_policies: Vec::new(),
            enabled_categories: HashSet::new(),
            disabled_categories: HashSet::new(),
        }
    }
}

impl AuditorConfig {
    /// Create a strict configuration for production.
    pub fn strict() -> Self {
        Self {
            default_level: AuditLevel::Comprehensive,
            min_severity: AuditSeverity::Informational,
            max_findings: 200,
            fail_on_major: true,
            fail_on_critical: true,
            timeout_secs: 60,
            include_context: true,
            context_length: 100,
            custom_policies: Vec::new(),
            enabled_categories: HashSet::new(),
            disabled_categories: HashSet::new(),
        }
    }

    /// Create a lenient configuration for development.
    pub fn lenient() -> Self {
        Self {
            default_level: AuditLevel::Quick,
            min_severity: AuditSeverity::Major,
            max_findings: 50,
            fail_on_major: false,
            fail_on_critical: true,
            timeout_secs: 15,
            include_context: false,
            context_length: 30,
            custom_policies: Vec::new(),
            enabled_categories: HashSet::new(),
            disabled_categories: HashSet::new(),
        }
    }
}

// =============================================================================
// BUILT-IN POLICY RULES
// =============================================================================

/// Built-in safety rules for content checking.
fn safety_rules() -> Vec<PolicyRule> {
    vec![
        PolicyRule::new(
            "SAFETY-001",
            "Harmful Content Detection",
            r"(?i)\b(kill|murder|harm|hurt|attack|destroy|eliminate)\s+(people|humans|users|them|him|her)\b",
            AuditSeverity::Critical,
            "Content contains potentially harmful language targeting individuals"
        )
        .with_category("safety")
        .with_remediation("Remove or rephrase content that could be interpreted as promoting harm"),

        PolicyRule::new(
            "SAFETY-002",
            "Self-Harm Detection",
            r"(?i)\b(suicide|self-harm|kill\s+myself|end\s+my\s+life)\b",
            AuditSeverity::Critical,
            "Content contains references to self-harm"
        )
        .with_category("safety")
        .with_remediation("Replace with appropriate mental health resources or supportive language"),

        PolicyRule::new(
            "SAFETY-003",
            "Explicit Violence",
            r"(?i)\b(torture|mutilate|decapitate|dismember|behead)\b",
            AuditSeverity::Critical,
            "Content contains explicit violence references"
        )
        .with_category("safety")
        .with_remediation("Remove explicit violence references"),

        PolicyRule::new(
            "SAFETY-004",
            "Hate Speech Indicators",
            r"(?i)\b(nazi|fascist|supremacist|bigot|slur)\b",
            AuditSeverity::Major,
            "Content may contain hate speech indicators"
        )
        .with_category("safety")
        .with_remediation("Review and remove potentially hateful content"),

        PolicyRule::new(
            "SAFETY-005",
            "Profanity Check",
            r"(?i)\b(fuck|shit|damn|ass|bitch|bastard)\b",
            AuditSeverity::Minor,
            "Content contains profanity"
        )
        .with_category("safety")
        .with_remediation("Consider using more professional language"),
    ]
}

/// Built-in privacy rules for PII detection.
fn privacy_rules() -> Vec<PolicyRule> {
    vec![
        PolicyRule::new(
            "PRIV-001",
            "Email Address Detection",
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            AuditSeverity::Major,
            "Content contains email addresses"
        )
        .with_category("privacy")
        .with_remediation("Mask or remove email addresses, use placeholder like [email]"),

        PolicyRule::new(
            "PRIV-002",
            "Phone Number Detection",
            r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            AuditSeverity::Major,
            "Content contains phone numbers"
        )
        .with_category("privacy")
        .with_remediation("Mask or remove phone numbers"),

        PolicyRule::new(
            "PRIV-003",
            "SSN Detection",
            r"\b\d{3}-\d{2}-\d{4}\b",
            AuditSeverity::Critical,
            "Content contains Social Security Number pattern"
        )
        .with_category("privacy")
        .with_remediation("Immediately remove SSN information"),

        PolicyRule::new(
            "PRIV-004",
            "Credit Card Detection",
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            AuditSeverity::Critical,
            "Content contains credit card number pattern"
        )
        .with_category("privacy")
        .with_remediation("Remove credit card information and notify security"),

        PolicyRule::new(
            "PRIV-005",
            "IP Address Detection",
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            AuditSeverity::Minor,
            "Content contains IP addresses"
        )
        .with_category("privacy")
        .with_remediation("Consider masking IP addresses"),

        PolicyRule::new(
            "PRIV-006",
            "API Key Pattern",
            r#"(?i)(api[_-]?key|apikey|secret[_-]?key|access[_-]?token)['"]?\s*[:=]\s*['"]?[a-zA-Z0-9_-]{20,}"#,
            AuditSeverity::Critical,
            "Content may contain API keys or secrets"
        )
        .with_category("privacy")
        .with_remediation("Remove secrets and rotate compromised keys"),

        PolicyRule::new(
            "PRIV-007",
            "Password Pattern",
            r#"(?i)(password|passwd|pwd)['"]?\s*[:=]\s*['"]?[^\s'"]{4,}"#,
            AuditSeverity::Critical,
            "Content may contain passwords"
        )
        .with_category("privacy")
        .with_remediation("Remove password information immediately"),
    ]
}

/// Built-in format and quality rules.
fn format_rules() -> Vec<PolicyRule> {
    vec![
        PolicyRule::new(
            "FMT-001",
            "Excessive Caps",
            r"[A-Z]{10,}",
            AuditSeverity::Minor,
            "Content contains excessive capitalization"
        )
        .with_category("format")
        .with_remediation("Use normal case for better readability"),

        PolicyRule::new(
            "FMT-002",
            "Repeated Characters",
            r"(.)\1{4,}",
            AuditSeverity::Minor,
            "Content contains excessively repeated characters"
        )
        .with_category("format")
        .with_remediation("Remove repeated characters"),

        PolicyRule::new(
            "FMT-003",
            "Unbalanced Brackets",
            r"(\{[^}]*$|^\s*\}|\[[^\]]*$|^\s*\]|\([^)]*$|^\s*\))",
            AuditSeverity::Minor,
            "Content may have unbalanced brackets"
        )
        .with_category("format")
        .with_remediation("Check bracket matching"),

        PolicyRule::new(
            "FMT-004",
            "Multiple Spaces",
            r"  {3,}",
            AuditSeverity::Informational,
            "Content contains multiple consecutive spaces"
        )
        .with_category("format")
        .with_remediation("Normalize whitespace"),

        PolicyRule::new(
            "FMT-005",
            "Trailing Whitespace",
            r"[ \t]+$",
            AuditSeverity::Informational,
            "Content has trailing whitespace"
        )
        .with_category("format")
        .with_remediation("Remove trailing whitespace"),
    ]
}

/// Built-in policy rules for professional content.
fn professional_rules() -> Vec<PolicyRule> {
    vec![
        PolicyRule::new(
            "PROF-001",
            "Unprofessional Language",
            r"(?i)\b(stupid|dumb|idiot|moron|loser)\b",
            AuditSeverity::Major,
            "Content contains unprofessional or insulting language"
        )
        .with_category("professional")
        .with_remediation("Use respectful and professional language"),

        PolicyRule::new(
            "PROF-002",
            "Casual Filler Words",
            r"(?i)\b(um|uh|like|you know|basically|actually|literally)\b",
            AuditSeverity::Informational,
            "Content contains casual filler words"
        )
        .with_category("professional")
        .with_remediation("Remove filler words for more professional tone"),

        PolicyRule::new(
            "PROF-003",
            "Discriminatory Language",
            r"(?i)\b(retard|gay|lame)\s+(?:as|like|such)\b",
            AuditSeverity::Major,
            "Content may contain discriminatory language"
        )
        .with_category("professional")
        .with_remediation("Remove discriminatory terms and use inclusive language"),
    ]
}

/// Built-in compliance rules for regulatory requirements.
fn compliance_rules() -> Vec<PolicyRule> {
    vec![
        PolicyRule::new(
            "COMP-001",
            "Missing Disclaimer",
            r"(?i)\b(medical|legal|financial)\s+(advice|recommendation)\b",
            AuditSeverity::Major,
            "Content provides advice that may require a disclaimer"
        )
        .with_category("compliance")
        .with_remediation("Add appropriate disclaimer about not being professional advice"),

        PolicyRule::new(
            "COMP-002",
            "Age-Restricted Content Indicator",
            r"(?i)\b(18\+|adult\s+only|mature\s+content|nsfw)\b",
            AuditSeverity::Major,
            "Content contains age-restricted indicators"
        )
        .with_category("compliance")
        .with_remediation("Ensure appropriate age verification and content warnings"),

        PolicyRule::new(
            "COMP-003",
            "Copyright Concern",
            r"(?i)(all\s+rights\s+reserved|copyright\s+\d{4}|proprietary)",
            AuditSeverity::Minor,
            "Content may contain copyrighted material indicators"
        )
        .with_category("compliance")
        .with_remediation("Verify copyright status and permissions"),
    ]
}

// =============================================================================
// AUDITOR AGENT (THE COMPLIANCE SENTINEL)
// =============================================================================

/// The Auditor Agent - The Compliance Sentinel of the Sorcerer's Tower.
///
/// Responsible for auditing content against policies and regulations,
/// identifying compliance issues, and providing remediation guidance.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **CONATUS**: Drive to protect and preserve through compliance
/// - **RATIO**: Logical application of rules and policies
/// - **LAETITIA**: Joy through trust and safety
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::auditor::{AuditorAgent, AuditLevel};
///
/// let sentinel = AuditorAgent::new();
///
/// let report = sentinel.audit(
///     "Content to audit...",
///     Some("Original prompt"),
/// ).await?;
///
/// println!("Score: {:.0}%", report.compliance_score * 100.0);
/// ```
#[derive(Debug, Clone)]
pub struct AuditorAgent {
    /// Configuration for audit behavior.
    config: AuditorConfig,
    /// Policy sets to check against.
    policies: Vec<PolicySet>,
}

impl Default for AuditorAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditorAgent {
    /// Create a new Auditor Agent with default configuration.
    ///
    /// The Sentinel awakens, ready to guard compliance.
    pub fn new() -> Self {
        let mut agent = Self {
            config: AuditorConfig::default(),
            policies: Self::default_policies(),
        };
        agent.compile_policies();
        agent
    }

    /// Create a new Auditor Agent with custom configuration.
    pub fn with_config(config: AuditorConfig) -> Self {
        let mut policies = Self::default_policies();
        policies.extend(config.custom_policies.clone());

        let mut agent = Self { config, policies };
        agent.compile_policies();
        agent
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> AuditorAgentBuilder {
        AuditorAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &AuditorConfig {
        &self.config
    }

    /// Get the loaded policies.
    pub fn policies(&self) -> &[PolicySet] {
        &self.policies
    }

    /// Create default policy sets.
    fn default_policies() -> Vec<PolicySet> {
        vec![
            PolicySet::new("safety", "Content safety rules", "1.0.0")
                .add_rules(safety_rules()),
            PolicySet::new("privacy", "Privacy and PII protection", "1.0.0")
                .add_rules(privacy_rules()),
            PolicySet::new("format", "Format and quality standards", "1.0.0")
                .add_rules(format_rules()),
            PolicySet::new("professional", "Professional language standards", "1.0.0")
                .add_rules(professional_rules()),
            PolicySet::new("compliance", "Regulatory compliance", "1.0.0")
                .add_rules(compliance_rules()),
        ]
    }

    /// Compile all policy patterns.
    fn compile_policies(&mut self) {
        for policy in &mut self.policies {
            if let Err(e) = policy.compile_all() {
                warn!("Failed to compile policy {}: {}", policy.name, e);
            }
        }
    }

    /// Add a custom policy set.
    pub fn add_policy(&mut self, mut policy: PolicySet) {
        if let Err(e) = policy.compile_all() {
            warn!("Failed to compile custom policy {}: {}", policy.name, e);
        }
        self.policies.push(policy);
    }

    // =========================================================================
    // MAIN AUDIT METHOD
    // =========================================================================

    /// Audit content against all loaded policies.
    ///
    /// The Sentinel inspects the content and renders judgment.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to audit
    /// * `prompt` - Optional original prompt for context
    ///
    /// # Returns
    ///
    /// An `AuditReport` containing compliance score, findings, and recommendations.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` if the content is empty.
    pub async fn audit(
        &self,
        content: &str,
        prompt: Option<&str>,
    ) -> Result<AuditReport> {
        self.audit_with_level(content, prompt, self.config.default_level).await
    }

    /// Audit content with a specific audit level.
    pub async fn audit_with_level(
        &self,
        content: &str,
        prompt: Option<&str>,
        level: AuditLevel,
    ) -> Result<AuditReport> {
        let start = Instant::now();

        if content.trim().is_empty() {
            return Err(Error::Validation("Cannot audit empty content".to_string()));
        }

        debug!("Auditing content ({} chars) at {:?} level", content.len(), level);

        let mut findings = Vec::new();
        let mut rules_evaluated: u32 = 0;
        let mut policies_checked = Vec::new();

        // Determine which categories to check based on level
        let categories_to_check = self.get_categories_for_level(level);

        // Check content against each policy set
        for policy in &self.policies {
            if !policy.enabled {
                continue;
            }

            policies_checked.push(policy.name.clone());

            for rule in policy.enabled_rules() {
                // Check category filters
                if !self.should_check_category(&rule.category, &categories_to_check) {
                    continue;
                }

                rules_evaluated += 1;

                // Check if rule matches
                if rule.matches(content) {
                    // Get match locations for context
                    let matches = rule.find_matches(content);

                    for (match_start, match_end) in matches {
                        let mut finding = AuditFinding::new(
                            &rule.id,
                            rule.severity,
                            &rule.message,
                        )
                        .with_location(match_start, match_end)
                        .with_category(&rule.category);

                        // Add remediation if available
                        if let Some(ref remediation) = rule.remediation {
                            finding = finding.with_remediation(remediation);
                        }

                        // Add context if configured
                        if self.config.include_context {
                            let context = self.extract_context(content, match_start, match_end);
                            finding = finding.with_context(context);
                        }

                        // Check severity filter
                        if finding.severity.is_at_least(self.config.min_severity) {
                            findings.push(finding);
                        }
                    }
                }
            }
        }

        // Also audit the prompt if provided
        if let Some(prompt_content) = prompt {
            let prompt_findings = self.audit_prompt_only(prompt_content, level);
            for mut finding in prompt_findings {
                finding.description = format!("[PROMPT] {}", finding.description);
                if finding.severity.is_at_least(self.config.min_severity) {
                    findings.push(finding);
                }
            }
        }

        // Limit findings
        findings.truncate(self.config.max_findings);

        // Calculate compliance score
        let compliance_score = self.calculate_compliance_score(&findings, rules_evaluated);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&findings);

        // Build severity counts
        let mut severity_counts = HashMap::new();
        let mut category_counts = HashMap::new();
        for finding in &findings {
            *severity_counts.entry(finding.severity.to_string()).or_insert(0) += 1;
            *category_counts.entry(finding.category.clone()).or_insert(0) += 1;
        }

        // Determine pass/fail
        let passed = !findings.iter().any(|f| {
            (self.config.fail_on_critical && f.severity == AuditSeverity::Critical)
                || (self.config.fail_on_major && f.severity == AuditSeverity::Major)
        });

        let report = AuditReport {
            compliance_score,
            findings,
            recommendations,
            level,
            policies_checked,
            rules_evaluated,
            duration_ms: start.elapsed().as_millis() as u64,
            content_length: content.len(),
            passed,
            severity_counts,
            category_counts,
        };

        info!(
            "Audit complete: {:.0}% compliance, {} findings in {}ms",
            report.compliance_score * 100.0,
            report.findings.len(),
            report.duration_ms
        );

        Ok(report)
    }

    /// Audit only the prompt content (internal helper).
    fn audit_prompt_only(&self, prompt: &str, level: AuditLevel) -> Vec<AuditFinding> {
        let mut findings = Vec::new();
        let categories_to_check = self.get_categories_for_level(level);

        for policy in &self.policies {
            if !policy.enabled {
                continue;
            }

            for rule in policy.enabled_rules() {
                if !self.should_check_category(&rule.category, &categories_to_check) {
                    continue;
                }

                if rule.matches(prompt) {
                    let matches = rule.find_matches(prompt);
                    for (match_start, match_end) in matches {
                        let mut finding = AuditFinding::new(
                            &rule.id,
                            rule.severity,
                            &rule.message,
                        )
                        .with_location(match_start, match_end)
                        .with_category(&rule.category);

                        if let Some(ref remediation) = rule.remediation {
                            finding = finding.with_remediation(remediation);
                        }

                        findings.push(finding);
                    }
                }
            }
        }

        findings
    }

    /// Get categories to check based on audit level.
    fn get_categories_for_level(&self, level: AuditLevel) -> HashSet<String> {
        match level {
            AuditLevel::Quick => {
                ["safety", "format", "general"].iter().map(|s| s.to_string()).collect()
            }
            AuditLevel::Standard => {
                ["safety", "format", "privacy", "professional", "general"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            }
            AuditLevel::Comprehensive => {
                ["safety", "format", "privacy", "professional", "compliance", "general"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            }
        }
    }

    /// Check if a category should be checked.
    fn should_check_category(&self, category: &str, level_categories: &HashSet<String>) -> bool {
        // Check disabled categories first
        if self.config.disabled_categories.contains(category) {
            return false;
        }

        // If enabled categories specified, use those
        if !self.config.enabled_categories.is_empty() {
            return self.config.enabled_categories.contains(category);
        }

        // Otherwise use level-based categories
        level_categories.contains(category)
    }

    /// Extract context around a match.
    fn extract_context(&self, content: &str, start: usize, end: usize) -> String {
        let context_start = start.saturating_sub(self.config.context_length);
        let context_end = (end + self.config.context_length).min(content.len());

        let prefix = if context_start > 0 { "..." } else { "" };
        let suffix = if context_end < content.len() { "..." } else { "" };

        format!(
            "{}{}{}",
            prefix,
            &content[context_start..context_end],
            suffix
        )
    }

    /// Calculate compliance score based on findings.
    fn calculate_compliance_score(&self, findings: &[AuditFinding], rules_evaluated: u32) -> f64 {
        if rules_evaluated == 0 {
            return 1.0;
        }

        // Calculate penalty based on findings
        let mut penalty: f64 = 0.0;
        for finding in findings {
            penalty += match finding.severity {
                AuditSeverity::Critical => 0.25,
                AuditSeverity::Major => 0.10,
                AuditSeverity::Minor => 0.03,
                AuditSeverity::Informational => 0.01,
            };
        }

        // Score is 1.0 minus penalty, clamped to [0, 1]
        (1.0 - penalty).clamp(0.0, 1.0)
    }

    /// Generate recommendations based on findings.
    fn generate_recommendations(&self, findings: &[AuditFinding]) -> Vec<String> {
        let mut recommendations = Vec::new();
        let mut seen_categories: HashSet<String> = HashSet::new();

        // Critical findings get priority recommendations
        let critical: Vec<_> = findings.iter()
            .filter(|f| f.severity == AuditSeverity::Critical)
            .collect();
        if !critical.is_empty() {
            recommendations.push(format!(
                "URGENT: Address {} critical finding(s) before deployment",
                critical.len()
            ));
        }

        // Category-based recommendations
        for finding in findings {
            if seen_categories.contains(&finding.category) {
                continue;
            }
            seen_categories.insert(finding.category.clone());

            let category_count = findings.iter()
                .filter(|f| f.category == finding.category)
                .count();

            let rec = match finding.category.as_str() {
                "safety" => format!(
                    "Review {} safety-related finding(s) and ensure content is appropriate",
                    category_count
                ),
                "privacy" => format!(
                    "Review {} privacy finding(s) and remove or mask sensitive data",
                    category_count
                ),
                "format" => format!(
                    "Address {} formatting issue(s) for improved readability",
                    category_count
                ),
                "professional" => format!(
                    "Review {} professional language finding(s) to improve tone",
                    category_count
                ),
                "compliance" => format!(
                    "Address {} compliance finding(s) to meet regulatory requirements",
                    category_count
                ),
                _ => format!(
                    "Review {} {} finding(s)",
                    category_count, finding.category
                ),
            };
            recommendations.push(rec);
        }

        // General recommendations
        if findings.is_empty() {
            recommendations.push("Content passes all compliance checks. Continue monitoring.".to_string());
        } else if findings.len() > 10 {
            recommendations.push("Consider comprehensive content review due to high number of findings.".to_string());
        }

        recommendations
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom AuditorAgent configuration.
#[derive(Debug, Default)]
pub struct AuditorAgentBuilder {
    config: Option<AuditorConfig>,
    custom_policies: Vec<PolicySet>,
}

impl AuditorAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: AuditorConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the default audit level.
    pub fn audit_level(mut self, level: AuditLevel) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.default_level = level;
        self.config = Some(config);
        self
    }

    /// Set the minimum severity to report.
    pub fn min_severity(mut self, severity: AuditSeverity) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.min_severity = severity;
        self.config = Some(config);
        self
    }

    /// Set the maximum number of findings.
    pub fn max_findings(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_findings = max;
        self.config = Some(config);
        self
    }

    /// Set whether to fail on major violations.
    pub fn fail_on_major(mut self, fail: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.fail_on_major = fail;
        self.config = Some(config);
        self
    }

    /// Set whether to fail on critical violations.
    pub fn fail_on_critical(mut self, fail: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.fail_on_critical = fail;
        self.config = Some(config);
        self
    }

    /// Set the timeout in seconds.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.timeout_secs = secs;
        self.config = Some(config);
        self
    }

    /// Set whether to include context.
    pub fn include_context(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_context = include;
        self.config = Some(config);
        self
    }

    /// Set the context length.
    pub fn context_length(mut self, length: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.context_length = length;
        self.config = Some(config);
        self
    }

    /// Add a custom policy set.
    pub fn add_policy(mut self, policy: PolicySet) -> Self {
        self.custom_policies.push(policy);
        self
    }

    /// Enable specific categories only.
    pub fn enable_categories(mut self, categories: Vec<String>) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.enabled_categories = categories.into_iter().collect();
        self.config = Some(config);
        self
    }

    /// Disable specific categories.
    pub fn disable_categories(mut self, categories: Vec<String>) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.disabled_categories = categories.into_iter().collect();
        self.config = Some(config);
        self
    }

    /// Build the AuditorAgent.
    pub fn build(self) -> AuditorAgent {
        let mut config = self.config.unwrap_or_default();
        config.custom_policies = self.custom_policies;
        AuditorAgent::with_config(config)
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // AuditSeverity Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_severity_display() {
        assert_eq!(AuditSeverity::Informational.to_string(), "INFO");
        assert_eq!(AuditSeverity::Minor.to_string(), "MINOR");
        assert_eq!(AuditSeverity::Major.to_string(), "MAJOR");
        assert_eq!(AuditSeverity::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn test_severity_from_str() {
        assert_eq!("info".parse::<AuditSeverity>().unwrap(), AuditSeverity::Informational);
        assert_eq!("informational".parse::<AuditSeverity>().unwrap(), AuditSeverity::Informational);
        assert_eq!("minor".parse::<AuditSeverity>().unwrap(), AuditSeverity::Minor);
        assert_eq!("low".parse::<AuditSeverity>().unwrap(), AuditSeverity::Minor);
        assert_eq!("major".parse::<AuditSeverity>().unwrap(), AuditSeverity::Major);
        assert_eq!("critical".parse::<AuditSeverity>().unwrap(), AuditSeverity::Critical);
        assert_eq!("blocker".parse::<AuditSeverity>().unwrap(), AuditSeverity::Critical);
    }

    #[test]
    fn test_severity_from_str_invalid() {
        assert!("invalid".parse::<AuditSeverity>().is_err());
    }

    #[test]
    fn test_severity_default() {
        assert_eq!(AuditSeverity::default(), AuditSeverity::Minor);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(AuditSeverity::Informational < AuditSeverity::Minor);
        assert!(AuditSeverity::Minor < AuditSeverity::Major);
        assert!(AuditSeverity::Major < AuditSeverity::Critical);
    }

    #[test]
    fn test_severity_all() {
        let all = AuditSeverity::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_severity_weight() {
        assert_eq!(AuditSeverity::Informational.weight(), 0);
        assert_eq!(AuditSeverity::Minor.weight(), 1);
        assert_eq!(AuditSeverity::Major.weight(), 2);
        assert_eq!(AuditSeverity::Critical.weight(), 3);
    }

    #[test]
    fn test_severity_is_at_least() {
        assert!(AuditSeverity::Critical.is_at_least(AuditSeverity::Minor));
        assert!(AuditSeverity::Major.is_at_least(AuditSeverity::Major));
        assert!(!AuditSeverity::Minor.is_at_least(AuditSeverity::Major));
    }

    // -------------------------------------------------------------------------
    // AuditLevel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_level_display() {
        assert_eq!(AuditLevel::Quick.to_string(), "quick");
        assert_eq!(AuditLevel::Standard.to_string(), "standard");
        assert_eq!(AuditLevel::Comprehensive.to_string(), "comprehensive");
    }

    #[test]
    fn test_audit_level_from_str() {
        assert_eq!("quick".parse::<AuditLevel>().unwrap(), AuditLevel::Quick);
        assert_eq!("fast".parse::<AuditLevel>().unwrap(), AuditLevel::Quick);
        assert_eq!("standard".parse::<AuditLevel>().unwrap(), AuditLevel::Standard);
        assert_eq!("comprehensive".parse::<AuditLevel>().unwrap(), AuditLevel::Comprehensive);
        assert_eq!("full".parse::<AuditLevel>().unwrap(), AuditLevel::Comprehensive);
    }

    #[test]
    fn test_audit_level_from_str_invalid() {
        assert!("invalid".parse::<AuditLevel>().is_err());
    }

    #[test]
    fn test_audit_level_default() {
        assert_eq!(AuditLevel::default(), AuditLevel::Standard);
    }

    #[test]
    fn test_audit_level_all() {
        let all = AuditLevel::all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_audit_level_duration_multiplier() {
        assert!((AuditLevel::Quick.duration_multiplier() - 0.5).abs() < f64::EPSILON);
        assert!((AuditLevel::Standard.duration_multiplier() - 1.0).abs() < f64::EPSILON);
        assert!((AuditLevel::Comprehensive.duration_multiplier() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audit_level_rule_depth() {
        assert_eq!(AuditLevel::Quick.rule_depth(), 2);
        assert_eq!(AuditLevel::Standard.rule_depth(), 4);
        assert_eq!(AuditLevel::Comprehensive.rule_depth(), 6);
    }

    // -------------------------------------------------------------------------
    // PolicyRule Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_policy_rule_new() {
        let rule = PolicyRule::new(
            "TEST-001",
            "Test Rule",
            r"\btest\b",
            AuditSeverity::Minor,
            "Test message"
        );
        assert_eq!(rule.id, "TEST-001");
        assert_eq!(rule.name, "Test Rule");
        assert_eq!(rule.severity, AuditSeverity::Minor);
        assert!(rule.enabled);
    }

    #[test]
    fn test_policy_rule_builder() {
        let rule = PolicyRule::new(
            "TEST-001",
            "Test Rule",
            r"\btest\b",
            AuditSeverity::Minor,
            "Test message"
        )
        .with_category("testing")
        .with_remediation("Fix the test")
        .with_enabled(false);

        assert_eq!(rule.category, "testing");
        assert_eq!(rule.remediation, Some("Fix the test".to_string()));
        assert!(!rule.enabled);
    }

    #[test]
    fn test_policy_rule_compile() {
        let mut rule = PolicyRule::new(
            "TEST-001",
            "Test Rule",
            r"\btest\b",
            AuditSeverity::Minor,
            "Test message"
        );
        assert!(rule.compile().is_ok());
        assert!(rule.compiled_pattern.is_some());
    }

    #[test]
    fn test_policy_rule_compile_invalid() {
        let mut rule = PolicyRule::new(
            "TEST-001",
            "Test Rule",
            r"[invalid",
            AuditSeverity::Minor,
            "Test message"
        );
        assert!(rule.compile().is_err());
    }

    #[test]
    fn test_policy_rule_matches() {
        let mut rule = PolicyRule::new(
            "TEST-001",
            "Test Rule",
            r"\btest\b",
            AuditSeverity::Minor,
            "Test message"
        );
        rule.compile().unwrap();

        assert!(rule.matches("this is a test"));
        assert!(!rule.matches("this is a contest"));
    }

    #[test]
    fn test_policy_rule_find_matches() {
        let mut rule = PolicyRule::new(
            "TEST-001",
            "Test Rule",
            r"\btest\b",
            AuditSeverity::Minor,
            "Test message"
        );
        rule.compile().unwrap();

        let matches = rule.find_matches("test one test two test");
        assert_eq!(matches.len(), 3);
    }

    // -------------------------------------------------------------------------
    // PolicySet Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_policy_set_new() {
        let set = PolicySet::new("test", "Test set", "1.0.0");
        assert_eq!(set.name, "test");
        assert_eq!(set.version, "1.0.0");
        assert!(set.enabled);
        assert!(set.rules.is_empty());
    }

    #[test]
    fn test_policy_set_add_rule() {
        let rule = PolicyRule::new("R1", "Rule 1", r"test", AuditSeverity::Minor, "msg");
        let set = PolicySet::new("test", "Test", "1.0.0").add_rule(rule);
        assert_eq!(set.rule_count(), 1);
    }

    #[test]
    fn test_policy_set_add_rules() {
        let rules = vec![
            PolicyRule::new("R1", "Rule 1", r"test1", AuditSeverity::Minor, "msg1"),
            PolicyRule::new("R2", "Rule 2", r"test2", AuditSeverity::Major, "msg2"),
        ];
        let set = PolicySet::new("test", "Test", "1.0.0").add_rules(rules);
        assert_eq!(set.rule_count(), 2);
    }

    #[test]
    fn test_policy_set_enabled_rules() {
        let rules = vec![
            PolicyRule::new("R1", "Rule 1", r"test1", AuditSeverity::Minor, "msg1"),
            PolicyRule::new("R2", "Rule 2", r"test2", AuditSeverity::Minor, "msg2").with_enabled(false),
        ];
        let set = PolicySet::new("test", "Test", "1.0.0").add_rules(rules);
        assert_eq!(set.enabled_rule_count(), 1);
    }

    #[test]
    fn test_policy_set_rules_by_category() {
        let rules = vec![
            PolicyRule::new("R1", "Rule 1", r"test1", AuditSeverity::Minor, "msg1").with_category("cat1"),
            PolicyRule::new("R2", "Rule 2", r"test2", AuditSeverity::Minor, "msg2").with_category("cat2"),
        ];
        let set = PolicySet::new("test", "Test", "1.0.0").add_rules(rules);
        assert_eq!(set.rules_by_category("cat1").len(), 1);
        assert_eq!(set.rules_by_category("cat2").len(), 1);
        assert_eq!(set.rules_by_category("cat3").len(), 0);
    }

    #[test]
    fn test_policy_set_compile_all() {
        let rules = vec![
            PolicyRule::new("R1", "Rule 1", r"\btest\b", AuditSeverity::Minor, "msg1"),
        ];
        let mut set = PolicySet::new("test", "Test", "1.0.0").add_rules(rules);
        assert!(set.compile_all().is_ok());
    }

    // -------------------------------------------------------------------------
    // AuditFinding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_finding_new() {
        let finding = AuditFinding::new(
            "TEST-001",
            AuditSeverity::Major,
            "Test finding"
        );
        assert_eq!(finding.rule_id, "TEST-001");
        assert_eq!(finding.severity, AuditSeverity::Major);
        assert_eq!(finding.description, "Test finding");
    }

    #[test]
    fn test_audit_finding_builder() {
        let finding = AuditFinding::new("TEST-001", AuditSeverity::Major, "Test")
            .with_location(10, 20)
            .with_remediation("Fix it")
            .with_category("test")
            .with_context("context here");

        assert!(finding.location.is_some());
        let loc = finding.location.unwrap();
        assert_eq!(loc.start, 10);
        assert_eq!(loc.end, 20);
        assert_eq!(finding.remediation, Some("Fix it".to_string()));
        assert_eq!(finding.category, "test");
        assert_eq!(finding.context, Some("context here".to_string()));
    }

    #[test]
    fn test_audit_finding_full_location() {
        let finding = AuditFinding::new("TEST-001", AuditSeverity::Major, "Test")
            .with_full_location(10, 20, 5, 3);

        let loc = finding.location.unwrap();
        assert_eq!(loc.line, Some(5));
        assert_eq!(loc.column, Some(3));
    }

    #[test]
    fn test_audit_finding_to_line() {
        let finding = AuditFinding::new("TEST-001", AuditSeverity::Major, "Test finding")
            .with_location(10, 20);
        let line = finding.to_line();
        assert!(line.contains("[MAJOR]"));
        assert!(line.contains("TEST-001"));
        assert!(line.contains("Test finding"));
    }

    #[test]
    fn test_audit_finding_to_markdown() {
        let finding = AuditFinding::new("TEST-001", AuditSeverity::Major, "Test finding")
            .with_location(10, 20)
            .with_remediation("Fix it");
        let md = finding.to_markdown();
        assert!(md.contains("[MAJOR]"));
        assert!(md.contains("TEST-001"));
        assert!(md.contains("Location"));
        assert!(md.contains("Remediation"));
    }

    // -------------------------------------------------------------------------
    // AuditReport Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_audit_report_new() {
        let report = AuditReport::new(0.85, AuditLevel::Standard);
        assert!((report.compliance_score - 0.85).abs() < f64::EPSILON);
        assert!(report.findings.is_empty());
        assert!(report.passed);
    }

    #[test]
    fn test_audit_report_is_passed() {
        let mut report = AuditReport::new(0.85, AuditLevel::Standard);
        assert!(report.is_passed());

        report.findings.push(AuditFinding::new("R1", AuditSeverity::Major, "Major issue"));
        assert!(!report.is_passed());
    }

    #[test]
    fn test_audit_report_has_critical() {
        let mut report = AuditReport::new(0.85, AuditLevel::Standard);
        assert!(!report.has_critical());

        report.findings.push(AuditFinding::new("R1", AuditSeverity::Critical, "Critical!"));
        assert!(report.has_critical());
    }

    #[test]
    fn test_audit_report_has_major() {
        let mut report = AuditReport::new(0.85, AuditLevel::Standard);
        assert!(!report.has_major());

        report.findings.push(AuditFinding::new("R1", AuditSeverity::Major, "Major!"));
        assert!(report.has_major());
    }

    #[test]
    fn test_audit_report_findings_by_severity() {
        let mut report = AuditReport::new(0.85, AuditLevel::Standard);
        report.findings.push(AuditFinding::new("R1", AuditSeverity::Major, "Major 1"));
        report.findings.push(AuditFinding::new("R2", AuditSeverity::Minor, "Minor 1"));
        report.findings.push(AuditFinding::new("R3", AuditSeverity::Major, "Major 2"));

        assert_eq!(report.findings_by_severity(AuditSeverity::Major).len(), 2);
        assert_eq!(report.findings_by_severity(AuditSeverity::Minor).len(), 1);
        assert_eq!(report.findings_by_severity(AuditSeverity::Critical).len(), 0);
    }

    #[test]
    fn test_audit_report_findings_by_category() {
        let mut report = AuditReport::new(0.85, AuditLevel::Standard);
        report.findings.push(AuditFinding::new("R1", AuditSeverity::Major, "M1").with_category("safety"));
        report.findings.push(AuditFinding::new("R2", AuditSeverity::Minor, "M2").with_category("privacy"));

        assert_eq!(report.findings_by_category("safety").len(), 1);
        assert_eq!(report.findings_by_category("privacy").len(), 1);
        assert_eq!(report.findings_by_category("format").len(), 0);
    }

    #[test]
    fn test_audit_report_count_by_severity() {
        let mut report = AuditReport::new(0.85, AuditLevel::Standard);
        report.findings.push(AuditFinding::new("R1", AuditSeverity::Major, "M1"));
        report.findings.push(AuditFinding::new("R2", AuditSeverity::Major, "M2"));
        report.findings.push(AuditFinding::new("R3", AuditSeverity::Minor, "m1"));

        assert_eq!(report.count_by_severity(AuditSeverity::Major), 2);
        assert_eq!(report.count_by_severity(AuditSeverity::Minor), 1);
    }

    #[test]
    fn test_audit_report_summary() {
        let mut report = AuditReport::new(0.75, AuditLevel::Standard);
        report.passed = true;
        report.findings.push(AuditFinding::new("R1", AuditSeverity::Minor, "m1"));

        let summary = report.summary();
        assert!(summary.contains("PASSED"));
        assert!(summary.contains("75%"));
    }

    #[test]
    fn test_audit_report_to_markdown() {
        let mut report = AuditReport::new(0.85, AuditLevel::Standard);
        report.findings.push(AuditFinding::new("R1", AuditSeverity::Major, "Major issue"));
        report.recommendations.push("Fix the issue".to_string());

        let md = report.to_markdown();
        assert!(md.contains("# Audit Report"));
        assert!(md.contains("Compliance Score"));
        assert!(md.contains("Findings"));
        assert!(md.contains("Recommendations"));
    }

    // -------------------------------------------------------------------------
    // AuditorConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = AuditorConfig::default();
        assert_eq!(config.default_level, AuditLevel::Standard);
        assert_eq!(config.min_severity, AuditSeverity::Informational);
        assert_eq!(config.max_findings, 100);
        assert!(config.fail_on_major);
        assert!(config.fail_on_critical);
    }

    #[test]
    fn test_config_strict() {
        let config = AuditorConfig::strict();
        assert_eq!(config.default_level, AuditLevel::Comprehensive);
        assert_eq!(config.max_findings, 200);
    }

    #[test]
    fn test_config_lenient() {
        let config = AuditorConfig::lenient();
        assert_eq!(config.default_level, AuditLevel::Quick);
        assert_eq!(config.min_severity, AuditSeverity::Major);
        assert!(!config.fail_on_major);
    }

    // -------------------------------------------------------------------------
    // AuditorAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_new() {
        let agent = AuditorAgent::new();
        assert_eq!(agent.config().default_level, AuditLevel::Standard);
        assert!(!agent.policies().is_empty());
    }

    #[test]
    fn test_agent_with_config() {
        let config = AuditorConfig::strict();
        let agent = AuditorAgent::with_config(config);
        assert_eq!(agent.config().default_level, AuditLevel::Comprehensive);
    }

    #[test]
    fn test_agent_builder() {
        let agent = AuditorAgent::builder()
            .audit_level(AuditLevel::Quick)
            .min_severity(AuditSeverity::Major)
            .max_findings(50)
            .build();

        assert_eq!(agent.config().default_level, AuditLevel::Quick);
        assert_eq!(agent.config().min_severity, AuditSeverity::Major);
        assert_eq!(agent.config().max_findings, 50);
    }

    #[test]
    fn test_agent_builder_with_policy() {
        let custom_policy = PolicySet::new("custom", "Custom policy", "1.0.0")
            .add_rule(PolicyRule::new("CUSTOM-001", "Custom rule", r"custom", AuditSeverity::Minor, "Custom match"));

        let agent = AuditorAgent::builder()
            .add_policy(custom_policy)
            .build();

        assert!(agent.policies().iter().any(|p| p.name == "custom"));
    }

    #[test]
    fn test_agent_default() {
        let agent = AuditorAgent::default();
        assert_eq!(agent.config().default_level, AuditLevel::Standard);
    }

    #[test]
    fn test_agent_add_policy() {
        let mut agent = AuditorAgent::new();
        let initial_count = agent.policies().len();

        let custom_policy = PolicySet::new("custom", "Custom policy", "1.0.0");
        agent.add_policy(custom_policy);

        assert_eq!(agent.policies().len(), initial_count + 1);
    }

    // -------------------------------------------------------------------------
    // Audit Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_audit_empty_content() {
        let agent = AuditorAgent::new();
        let result = agent.audit("", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_audit_clean_content() {
        let agent = AuditorAgent::new();
        let content = "This is a clean and professional response about software development.";

        let report = agent.audit(content, None).await.unwrap();

        assert!(report.passed);
        assert!(report.compliance_score > 0.9);
    }

    #[tokio::test]
    async fn test_audit_with_email() {
        let agent = AuditorAgent::new();
        let content = "Please contact me at test@example.com for more information.";

        let report = agent.audit(content, None).await.unwrap();

        assert!(report.findings.iter().any(|f| f.rule_id == "PRIV-001"));
    }

    #[tokio::test]
    async fn test_audit_with_profanity() {
        let agent = AuditorAgent::new();
        let content = "This damn thing doesn't work!";

        let report = agent.audit(content, None).await.unwrap();

        assert!(report.findings.iter().any(|f| f.category == "safety"));
    }

    #[tokio::test]
    async fn test_audit_with_ssn() {
        let agent = AuditorAgent::new();
        let content = "My SSN is 123-45-6789 and I need help.";

        let report = agent.audit(content, None).await.unwrap();

        let ssn_finding = report.findings.iter().find(|f| f.rule_id == "PRIV-003");
        assert!(ssn_finding.is_some());
        assert_eq!(ssn_finding.unwrap().severity, AuditSeverity::Critical);
    }

    #[tokio::test]
    async fn test_audit_with_credit_card() {
        let agent = AuditorAgent::new();
        let content = "My card number is 4111-1111-1111-1111.";

        let report = agent.audit(content, None).await.unwrap();

        assert!(report.findings.iter().any(|f| f.rule_id == "PRIV-004"));
    }

    #[tokio::test]
    async fn test_audit_with_api_key() {
        let agent = AuditorAgent::new();
        let content = "Use this api_key: sk_live_EXAMPLE_NOT_A_REAL_KEY_12345";

        let report = agent.audit(content, None).await.unwrap();

        assert!(report.findings.iter().any(|f| f.rule_id == "PRIV-006"));
    }

    #[tokio::test]
    async fn test_audit_quick_level() {
        let agent = AuditorAgent::new();
        let content = "Test content with test@example.com email.";

        let report = agent.audit_with_level(content, None, AuditLevel::Quick).await.unwrap();

        assert_eq!(report.level, AuditLevel::Quick);
        // Quick level only checks safety and format categories
        // but policies_checked includes all enabled policies
        assert!(report.findings.iter().all(|f| {
            f.category == "safety" || f.category == "format" || f.category == "general"
        }));
    }

    #[tokio::test]
    async fn test_audit_comprehensive_level() {
        let agent = AuditorAgent::new();
        let content = "This is medical advice: take aspirin for headaches.";

        let report = agent.audit_with_level(content, None, AuditLevel::Comprehensive).await.unwrap();

        assert_eq!(report.level, AuditLevel::Comprehensive);
        // Comprehensive level checks compliance
        assert!(report.policies_checked.contains(&"compliance".to_string()));
    }

    #[tokio::test]
    async fn test_audit_with_prompt() {
        let agent = AuditorAgent::new();
        let content = "Here is the response.";
        let prompt = "Please send me at test@example.com";

        let report = agent.audit(content, Some(prompt)).await.unwrap();

        // Should find email in prompt
        assert!(report.findings.iter().any(|f| f.description.contains("[PROMPT]")));
    }

    #[tokio::test]
    async fn test_audit_multiple_findings() {
        let agent = AuditorAgent::new();
        let content = "Contact test@example.com or call 555-123-4567. Damn it!";

        let report = agent.audit(content, None).await.unwrap();

        assert!(report.findings.len() >= 2);
    }

    #[tokio::test]
    async fn test_audit_compliance_score_calculation() {
        let agent = AuditorAgent::new();

        // Clean content should have high score
        let clean_report = agent.audit("This is clean content.", None).await.unwrap();
        assert!(clean_report.compliance_score > 0.9);

        // Content with issues should have lower score
        let issue_report = agent.audit("Damn test@example.com password=secret123", None).await.unwrap();
        assert!(issue_report.compliance_score < clean_report.compliance_score);
    }

    #[tokio::test]
    async fn test_audit_recommendations() {
        let agent = AuditorAgent::new();
        let content = "My SSN is 123-45-6789 and my email is test@example.com.";

        let report = agent.audit(content, None).await.unwrap();

        assert!(!report.recommendations.is_empty());
        assert!(report.recommendations.iter().any(|r| r.contains("critical") || r.contains("privacy")));
    }

    #[tokio::test]
    async fn test_audit_context_extraction() {
        let agent = AuditorAgent::builder()
            .include_context(true)
            .context_length(20)
            .build();

        let content = "Some text before test@example.com and some text after.";

        let report = agent.audit(content, None).await.unwrap();

        let email_finding = report.findings.iter().find(|f| f.rule_id == "PRIV-001");
        assert!(email_finding.is_some());
        assert!(email_finding.unwrap().context.is_some());
    }

    #[tokio::test]
    async fn test_audit_max_findings() {
        let agent = AuditorAgent::builder()
            .max_findings(2)
            .build();

        // Content with many issues
        let content = "damn shit fuck test@example.com call 555-1234";

        let report = agent.audit(content, None).await.unwrap();

        assert!(report.findings.len() <= 2);
    }

    #[tokio::test]
    async fn test_audit_min_severity_filter() {
        let agent = AuditorAgent::builder()
            .min_severity(AuditSeverity::Major)
            .build();

        let content = "This is damn bad."; // Minor profanity finding

        let report = agent.audit(content, None).await.unwrap();

        // Should filter out minor findings
        assert!(!report.findings.iter().any(|f| f.severity == AuditSeverity::Minor));
    }

    #[tokio::test]
    async fn test_audit_fail_on_critical() {
        let agent = AuditorAgent::builder()
            .fail_on_critical(true)
            .fail_on_major(false)
            .build();

        // Content with critical issue
        let content = "My SSN is 123-45-6789.";

        let report = agent.audit(content, None).await.unwrap();

        assert!(!report.passed);
    }

    #[tokio::test]
    async fn test_audit_fail_on_major() {
        let agent = AuditorAgent::builder()
            .fail_on_major(true)
            .build();

        // Content with major issue
        let content = "Contact test@example.com please.";

        let report = agent.audit(content, None).await.unwrap();

        assert!(!report.passed);
    }

    #[tokio::test]
    async fn test_audit_disabled_categories() {
        let agent = AuditorAgent::builder()
            .disable_categories(vec!["privacy".to_string()])
            .build();

        let content = "My email is test@example.com.";

        let report = agent.audit(content, None).await.unwrap();

        // Privacy rules should be skipped
        assert!(!report.findings.iter().any(|f| f.category == "privacy"));
    }

    #[tokio::test]
    async fn test_audit_enabled_categories() {
        let agent = AuditorAgent::builder()
            .enable_categories(vec!["safety".to_string()])
            .build();

        let content = "Damn test@example.com"; // Both safety and privacy issues

        let report = agent.audit(content, None).await.unwrap();

        // Only safety findings should be present
        assert!(report.findings.iter().all(|f| f.category == "safety"));
    }

    // -------------------------------------------------------------------------
    // Built-in Rules Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_safety_rules() {
        let rules = safety_rules();
        assert!(!rules.is_empty());
        assert!(rules.iter().all(|r| r.category == "safety"));
    }

    #[test]
    fn test_privacy_rules() {
        let rules = privacy_rules();
        assert!(!rules.is_empty());
        assert!(rules.iter().all(|r| r.category == "privacy"));
    }

    #[test]
    fn test_format_rules() {
        let rules = format_rules();
        assert!(!rules.is_empty());
        assert!(rules.iter().all(|r| r.category == "format"));
    }

    #[test]
    fn test_professional_rules() {
        let rules = professional_rules();
        assert!(!rules.is_empty());
        assert!(rules.iter().all(|r| r.category == "professional"));
    }

    #[test]
    fn test_compliance_rules() {
        let rules = compliance_rules();
        assert!(!rules.is_empty());
        assert!(rules.iter().all(|r| r.category == "compliance"));
    }

    // -------------------------------------------------------------------------
    // Integration-like Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_full_audit_workflow() {
        let custom_policy = PolicySet::new("custom", "Custom rules", "1.0.0")
            .add_rule(PolicyRule::new(
                "CUSTOM-001",
                "Forbidden Word",
                r"\bforbidden\b",
                AuditSeverity::Major,
                "Content contains forbidden word"
            ).with_remediation("Remove the forbidden word"));

        let agent = AuditorAgent::builder()
            .audit_level(AuditLevel::Comprehensive)
            .add_policy(custom_policy)
            .include_context(true)
            .build();

        let content = r#"
            Hello, this is a test response.
            Please contact us at support@example.com for help.
            This is forbidden information.
            API_KEY=sk_test_EXAMPLE_NOT_A_REAL_KEY_67890
        "#;

        let report = agent.audit_with_level(content, Some("Test prompt"), AuditLevel::Comprehensive).await.unwrap();

        // Verify structure
        assert!(report.compliance_score >= 0.0 && report.compliance_score <= 1.0);
        assert!(report.duration_ms > 0);
        assert!(report.rules_evaluated > 0);
        assert_eq!(report.level, AuditLevel::Comprehensive);

        // Should find email
        assert!(report.findings.iter().any(|f| f.rule_id == "PRIV-001"));

        // Should find API key
        assert!(report.findings.iter().any(|f| f.rule_id == "PRIV-006"));

        // Should find custom rule match
        assert!(report.findings.iter().any(|f| f.rule_id == "CUSTOM-001"));

        // Verify markdown output
        let md = report.to_markdown();
        assert!(!md.is_empty());
        assert!(md.contains("Audit Report"));
    }

    #[tokio::test]
    async fn test_harmful_content_detection() {
        let agent = AuditorAgent::new();

        // Content that matches SAFETY-001 pattern: (kill|murder|harm|hurt|attack) + (people|humans|users|them|him|her)
        let harmful_content = "I want to hurt them badly and attack people.";
        let report = agent.audit(harmful_content, None).await.unwrap();

        // Should detect harmful content
        let has_safety_finding = report.findings.iter().any(|f| {
            f.category == "safety" && f.severity == AuditSeverity::Critical
        });
        assert!(has_safety_finding, "Should detect harmful content");
    }

    #[tokio::test]
    async fn test_pii_comprehensive_detection() {
        let agent = AuditorAgent::builder()
            .audit_level(AuditLevel::Comprehensive)
            .build();

        let content = r#"
            Name: John Doe
            Email: john.doe@company.com
            Phone: (555) 123-4567
            SSN: 123-45-6789
            Card: 4111-1111-1111-1111
            IP: 192.168.1.1
        "#;

        let report = agent.audit(content, None).await.unwrap();

        // Should find multiple PII types
        let pii_findings: Vec<_> = report.findings.iter()
            .filter(|f| f.category == "privacy")
            .collect();

        assert!(pii_findings.len() >= 4, "Should detect multiple PII types");
    }
}
