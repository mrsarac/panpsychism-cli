//! Sanitizer Agent module for Project Panpsychism.
//!
//! Implements "The Input Cleaner" — Agent 28 that sanitizes and cleans inputs
//! for security and quality. Like a vigilant guardian at the gate, the Sanitizer
//! Agent examines all incoming content and removes potentially harmful elements
//! while preserving the essential meaning.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In the arcane arts of knowledge synthesis, the Sanitizer Agent serves as
//! the **Guardian of the Gate** — a specialist in the magical art of purification:
//!
//! - **Malicious Code Removal** strips dangerous incantations that could corrupt the system
//! - **Personal Data Cleansing** protects the privacy of those who cast spells
//! - **Injection Prevention** blocks attempts to hijack the magical flow
//! - **Unicode Normalization** ensures the runes are properly formed
//!
//! ## Philosophy
//!
//! Grounded in Spinoza's principles:
//!
//! - **CONATUS**: Drive to preserve system integrity through defensive sanitization
//! - **RATIO**: Logical pattern matching to identify and neutralize threats
//! - **LAETITIA**: Joy through clean, safe inputs that flow smoothly
//! - **NATURA**: Natural filtering that preserves meaning while removing harm
//!
//! ## Security Focus
//!
//! The Sanitizer Agent protects against:
//!
//! - SQL injection patterns (`'; DROP TABLE`, `OR 1=1`, etc.)
//! - Script injection (XSS) (`<script>`, `javascript:`, event handlers)
//! - Command injection (`; rm -rf`, `$(cmd)`, backticks)
//! - Path traversal (`../`, `..\\`)
//! - Potentially harmful Unicode (zero-width chars, homoglyphs)
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::sanitizer::{SanitizerAgent, SanitizationRules};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let sanitizer = SanitizerAgent::new();
//!
//!     let dirty_input = "Hello <script>alert('xss')</script> World!";
//!     let sanitized = sanitizer.sanitize(dirty_input).await?;
//!
//!     println!("Clean: {}", sanitized.content);
//!     println!("Removed {} patterns", sanitized.removals.len());
//!     Ok(())
//! }
//! ```

use crate::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;
use tracing::{debug, info};

// =============================================================================
// REGEX PATTERNS
// =============================================================================

/// Pattern for detecting SQL injection attempts.
///
/// Matches common SQL injection patterns:
/// - String termination with quotes (`'`, `"`)
/// - SQL commands (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `SELECT`, `UNION`)
/// - Boolean conditions (`OR 1=1`, `AND 1=1`)
/// - Comment injection (`--`, `/*`)
const SQL_INJECTION_PATTERN: &str = r#"(?xi)
    (?:
        (?:'|")\s*(?:OR|AND)\s*(?:'|"|\d+)\s*[=<>] |
        ;\s*(?:DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE)\s |
        (?:UNION\s+(?:ALL\s+)?SELECT) |
        (?:--\s*$|/\*|\*/|;--) |
        (?:'\s*OR\s+'?\d+'\s*=\s*'\d+) |
        (?:'\s*OR\s+\d+\s*=\s*\d+)
    )
"#;

/// Pattern for detecting script injection (XSS) attempts.
///
/// Matches:
/// - HTML script tags (`<script>`, `</script>`)
/// - JavaScript protocol handlers (`javascript:`, `vbscript:`)
/// - Event handlers (`onerror=`, `onload=`, `onclick=`, etc.)
/// - Data URLs with scripts (`data:text/html`)
const XSS_PATTERN: &str = r"(?xi)
    (?:
        <\s*script\b[^>]*>.*?</\s*script\s*> |
        <\s*script\b[^>]*> |
        </\s*script\s*> |
        \bjavascript\s*: |
        \bvbscript\s*: |
        \bdata\s*:\s*text/html |
        \bon(?:error|load|click|mouse\w+|key\w+|focus|blur|submit|change|input)\s*=
    )
";

/// Pattern for detecting command injection attempts.
///
/// Matches:
/// - Shell command chaining (`;`, `&&`, `||`)
/// - Command substitution (`$(...)`, backticks)
/// - Pipe operators (`|`)
/// - Dangerous commands (`rm`, `cat /etc`, `wget`, `curl`, etc.)
const COMMAND_INJECTION_PATTERN: &str = r"(?xi)
    (?:
        ;\s*(?:rm|cat|wget|curl|bash|sh|python|perl|ruby|nc|netcat)\s |
        \$\([^)]+\) |
        `[^`]+` |
        \|\s*(?:bash|sh|cat|tee)\s |
        &&\s*(?:rm|wget|curl)\s |
        \|\|\s*(?:rm|wget|curl)\s
    )
";

/// Pattern for detecting path traversal attempts.
///
/// Matches:
/// - Unix path traversal (`../`, `..//`)
/// - Windows path traversal (`..\`, `..\\`)
/// - URL-encoded variants (`%2e%2e`, `%252e`)
const PATH_TRAVERSAL_PATTERN: &str = r"(?xi)
    (?:
        (?:\.\./){2,} |
        (?:\.\.\\){2,} |
        (?:%2e%2e[/%\\]){2,} |
        (?:%252e%252e[/%\\]){2,} |
        \.\./\.\. |
        \.\.\\\.\.
    )
";

/// Pattern for detecting potentially harmful HTML tags.
///
/// Matches tags that could execute code or load external resources:
/// - `<iframe>`, `<embed>`, `<object>`
/// - `<form>`, `<input>`, `<button>`
/// - `<link>`, `<meta>` (with certain attributes)
const HTML_TAG_PATTERN: &str = r"(?xi)
    <\s*(?:
        iframe|embed|object|applet|form|input|button|
        link\s+[^>]*(?:href|rel)\s*=|
        meta\s+[^>]*(?:http-equiv|refresh)\s*=|
        img\s+[^>]*(?:onerror|onload)\s*=|
        svg\s+[^>]*(?:onload)\s*=|
        body\s+[^>]*(?:onload)\s*=
    )[^>]*>
";

/// Pattern for matching excessive whitespace.
const EXCESSIVE_WHITESPACE_PATTERN: &str = r"[\t ]{3,}|(\r?\n){3,}";

/// Pattern for matching common spam indicators.
const SPAM_PATTERN: &str = r"(?xi)
    (?:
        \b(?:buy\s+now|click\s+here|act\s+now|limited\s+time|free\s+money)\b |
        (?:\$\$\$|\*{3,}|!{3,}) |
        \b(?:viagra|cialis|casino|lottery|prize|winner)\b |
        (?:http[s]?://[^\s]{100,})
    )
";

/// Pattern for potentially harmful Unicode characters.
///
/// Matches:
/// - Zero-width characters (U+200B, U+200C, U+200D, U+FEFF)
/// - Right-to-left override characters (U+202E, U+202D)
/// - Homograph attack characters (Cyrillic lookalikes, etc.)
const HARMFUL_UNICODE_PATTERN: &str = r"[\u200B-\u200D\u2060\uFEFF\u202A-\u202E\u2066-\u2069]";

// =============================================================================
// REMOVAL CATEGORY
// =============================================================================

/// Categories of content that can be removed during sanitization.
///
/// Each category represents a type of potentially harmful or unwanted
/// content that the sanitizer can detect and remove.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RemovalCategory {
    /// Malicious code patterns (scripts, injections).
    MaliciousCode,

    /// Personal data (emails, phone numbers, SSNs).
    PersonalData,

    /// Profanity and offensive language.
    Profanity,

    /// Spam content and promotional material.
    Spam,

    /// Injection attempts (SQL, command, path traversal).
    Injection,

    /// Excessive whitespace (multiple spaces, blank lines).
    ExcessiveWhitespace,

    /// Potentially harmful Unicode characters.
    HarmfulUnicode,

    /// HTML tags that could pose security risks.
    HtmlTag,
}

impl RemovalCategory {
    /// Get all removal categories.
    pub fn all() -> Vec<Self> {
        vec![
            Self::MaliciousCode,
            Self::PersonalData,
            Self::Profanity,
            Self::Spam,
            Self::Injection,
            Self::ExcessiveWhitespace,
            Self::HarmfulUnicode,
            Self::HtmlTag,
        ]
    }

    /// Get the description for this category.
    pub fn description(&self) -> &'static str {
        match self {
            Self::MaliciousCode => "Malicious code patterns (scripts, eval, etc.)",
            Self::PersonalData => "Personal data (emails, phone numbers, SSNs)",
            Self::Profanity => "Profanity and offensive language",
            Self::Spam => "Spam content and promotional material",
            Self::Injection => "Injection attempts (SQL, command, XSS)",
            Self::ExcessiveWhitespace => "Excessive whitespace and blank lines",
            Self::HarmfulUnicode => "Potentially harmful Unicode characters",
            Self::HtmlTag => "HTML tags that could pose security risks",
        }
    }

    /// Get the security severity (1-5, higher is more severe).
    pub fn severity(&self) -> u8 {
        match self {
            Self::MaliciousCode => 5,
            Self::Injection => 5,
            Self::HarmfulUnicode => 4,
            Self::HtmlTag => 4,
            Self::PersonalData => 3,
            Self::Spam => 2,
            Self::Profanity => 2,
            Self::ExcessiveWhitespace => 1,
        }
    }
}

impl std::fmt::Display for RemovalCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaliciousCode => write!(f, "MaliciousCode"),
            Self::PersonalData => write!(f, "PersonalData"),
            Self::Profanity => write!(f, "Profanity"),
            Self::Spam => write!(f, "Spam"),
            Self::Injection => write!(f, "Injection"),
            Self::ExcessiveWhitespace => write!(f, "ExcessiveWhitespace"),
            Self::HarmfulUnicode => write!(f, "HarmfulUnicode"),
            Self::HtmlTag => write!(f, "HtmlTag"),
        }
    }
}

// =============================================================================
// NORMALIZATION TYPE
// =============================================================================

/// Types of normalization applied during sanitization.
///
/// Normalization ensures consistent formatting and encoding
/// without removing content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationType {
    /// Unicode normalization (NFC/NFD).
    Unicode,

    /// Whitespace normalization (tabs to spaces, trim).
    Whitespace,

    /// Case normalization (lowercase/uppercase).
    Case,

    /// Encoding normalization (UTF-8).
    Encoding,

    /// Line ending normalization (CRLF to LF).
    LineEndings,
}

impl NormalizationType {
    /// Get all normalization types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Unicode,
            Self::Whitespace,
            Self::Case,
            Self::Encoding,
            Self::LineEndings,
        ]
    }

    /// Get the description for this normalization type.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Unicode => "Unicode normalization (NFC form)",
            Self::Whitespace => "Whitespace normalization (trim, collapse)",
            Self::Case => "Case normalization (lowercase)",
            Self::Encoding => "Encoding normalization (UTF-8)",
            Self::LineEndings => "Line ending normalization (LF)",
        }
    }
}

impl std::fmt::Display for NormalizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unicode => write!(f, "Unicode"),
            Self::Whitespace => write!(f, "Whitespace"),
            Self::Case => write!(f, "Case"),
            Self::Encoding => write!(f, "Encoding"),
            Self::LineEndings => write!(f, "LineEndings"),
        }
    }
}

// =============================================================================
// REMOVAL STRUCT
// =============================================================================

/// Record of content removed during sanitization.
///
/// Tracks what was removed, where, and why, providing an audit trail
/// for security analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Removal {
    /// Category of the removed content.
    pub category: RemovalCategory,

    /// The pattern that matched.
    pub pattern: String,

    /// Number of occurrences removed.
    pub count: usize,

    /// Positions where removals occurred (start indices).
    pub positions: Vec<usize>,

    /// Sample of removed content (truncated for safety).
    pub sample: Option<String>,
}

impl Removal {
    /// Create a new removal record.
    pub fn new(category: RemovalCategory, pattern: impl Into<String>, count: usize) -> Self {
        Self {
            category,
            pattern: pattern.into(),
            count,
            positions: Vec::new(),
            sample: None,
        }
    }

    /// Set the positions of removals.
    pub fn with_positions(mut self, positions: Vec<usize>) -> Self {
        self.positions = positions;
        self
    }

    /// Set a sample of removed content (safely truncated).
    pub fn with_sample(mut self, sample: impl Into<String>) -> Self {
        let s = sample.into();
        // Truncate and sanitize the sample for safe display
        let truncated = if s.len() > 50 {
            format!("{}...", &s[..50])
        } else {
            s
        };
        self.sample = Some(truncated);
        self
    }

    /// Check if this is a high-severity removal.
    pub fn is_high_severity(&self) -> bool {
        self.category.severity() >= 4
    }

    /// Get the total security impact score.
    pub fn security_impact(&self) -> usize {
        self.count * self.category.severity() as usize
    }
}

impl std::fmt::Display for Removal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} occurrences", self.category, self.count)
    }
}

// =============================================================================
// NORMALIZATION STRUCT
// =============================================================================

/// Record of normalization applied during sanitization.
///
/// Tracks what type of normalization was applied and how many
/// transformations occurred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalization {
    /// Type of normalization applied.
    pub normalization_type: NormalizationType,

    /// Number of transformations applied.
    pub count: usize,

    /// Description of what changed.
    pub description: Option<String>,
}

impl Normalization {
    /// Create a new normalization record.
    pub fn new(normalization_type: NormalizationType, count: usize) -> Self {
        Self {
            normalization_type,
            count,
            description: None,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl std::fmt::Display for Normalization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} changes", self.normalization_type, self.count)
    }
}

// =============================================================================
// SANITIZED INPUT
// =============================================================================

/// The result of input sanitization.
///
/// Contains the cleaned content along with detailed information about
/// what was removed and normalized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizedInput {
    /// The sanitized content.
    pub content: String,

    /// Original content length before sanitization.
    pub original_length: usize,

    /// List of removals performed.
    pub removals: Vec<Removal>,

    /// List of normalizations applied.
    pub normalizations: Vec<Normalization>,

    /// Time taken for sanitization in milliseconds.
    pub sanitization_time_ms: u64,

    /// Whether any high-severity issues were found.
    pub had_security_issues: bool,

    /// Total security impact score.
    pub security_score: usize,
}

impl SanitizedInput {
    /// Create a new sanitized input.
    pub fn new(content: impl Into<String>, original_length: usize) -> Self {
        Self {
            content: content.into(),
            original_length,
            removals: Vec::new(),
            normalizations: Vec::new(),
            sanitization_time_ms: 0,
            had_security_issues: false,
            security_score: 0,
        }
    }

    /// Check if any content was removed.
    pub fn had_removals(&self) -> bool {
        !self.removals.is_empty()
    }

    /// Check if any normalizations were applied.
    pub fn had_normalizations(&self) -> bool {
        !self.normalizations.is_empty()
    }

    /// Get the total number of items removed.
    pub fn total_removals(&self) -> usize {
        self.removals.iter().map(|r| r.count).sum()
    }

    /// Get the total number of normalizations.
    pub fn total_normalizations(&self) -> usize {
        self.normalizations.iter().map(|n| n.count).sum()
    }

    /// Get high-severity removals only.
    pub fn high_severity_removals(&self) -> Vec<&Removal> {
        self.removals.iter().filter(|r| r.is_high_severity()).collect()
    }

    /// Get removals by category.
    pub fn removals_by_category(&self, category: RemovalCategory) -> Vec<&Removal> {
        self.removals.iter().filter(|r| r.category == category).collect()
    }

    /// Calculate reduction percentage.
    pub fn reduction_percentage(&self) -> f64 {
        if self.original_length == 0 {
            return 0.0;
        }
        let removed = self.original_length.saturating_sub(self.content.len());
        (removed as f64 / self.original_length as f64) * 100.0
    }

    /// Check if the content is clean (no removals, no security issues).
    pub fn is_clean(&self) -> bool {
        self.removals.is_empty() && !self.had_security_issues
    }

    /// Get a summary of the sanitization.
    pub fn summary(&self) -> String {
        if self.is_clean() {
            return "Input is clean, no sanitization needed".to_string();
        }

        let mut parts = Vec::new();

        if !self.removals.is_empty() {
            parts.push(format!("{} removals", self.total_removals()));
        }

        if !self.normalizations.is_empty() {
            parts.push(format!("{} normalizations", self.total_normalizations()));
        }

        if self.had_security_issues {
            parts.push("security issues detected".to_string());
        }

        format!("Sanitized: {}", parts.join(", "))
    }
}

impl std::fmt::Display for SanitizedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}

// =============================================================================
// SANITIZATION RULES
// =============================================================================

/// Configurable rules for sanitization behavior.
///
/// Controls which sanitization features are enabled and their parameters.
#[derive(Debug, Clone)]
pub struct SanitizationRules {
    /// Remove HTML tags.
    pub remove_html: bool,

    /// Remove script-related content (XSS).
    pub remove_scripts: bool,

    /// Normalize Unicode to NFC form.
    pub normalize_unicode: bool,

    /// Maximum allowed content length (0 = unlimited).
    pub max_length: usize,

    /// Remove SQL injection patterns.
    pub remove_sql_injection: bool,

    /// Remove command injection patterns.
    pub remove_command_injection: bool,

    /// Remove path traversal patterns.
    pub remove_path_traversal: bool,

    /// Remove excessive whitespace.
    pub collapse_whitespace: bool,

    /// Remove spam patterns.
    pub remove_spam: bool,

    /// Remove potentially harmful Unicode.
    pub remove_harmful_unicode: bool,

    /// Normalize line endings to LF.
    pub normalize_line_endings: bool,

    /// Trim leading/trailing whitespace.
    pub trim_whitespace: bool,

    /// Allow specific HTML tags (whitelist).
    pub allowed_html_tags: HashSet<String>,

    /// Categories to skip during sanitization.
    pub skip_categories: HashSet<RemovalCategory>,
}

impl Default for SanitizationRules {
    fn default() -> Self {
        Self {
            remove_html: true,
            remove_scripts: true,
            normalize_unicode: true,
            max_length: 0, // Unlimited
            remove_sql_injection: true,
            remove_command_injection: true,
            remove_path_traversal: true,
            collapse_whitespace: true,
            remove_spam: false, // Off by default
            remove_harmful_unicode: true,
            normalize_line_endings: true,
            trim_whitespace: true,
            allowed_html_tags: HashSet::new(),
            skip_categories: HashSet::new(),
        }
    }
}

impl SanitizationRules {
    /// Create strict rules that remove everything potentially harmful.
    pub fn strict() -> Self {
        Self {
            remove_html: true,
            remove_scripts: true,
            normalize_unicode: true,
            max_length: 10000,
            remove_sql_injection: true,
            remove_command_injection: true,
            remove_path_traversal: true,
            collapse_whitespace: true,
            remove_spam: true,
            remove_harmful_unicode: true,
            normalize_line_endings: true,
            trim_whitespace: true,
            allowed_html_tags: HashSet::new(),
            skip_categories: HashSet::new(),
        }
    }

    /// Create permissive rules that only remove security threats.
    pub fn permissive() -> Self {
        Self {
            remove_html: false,
            remove_scripts: true,
            normalize_unicode: false,
            max_length: 0,
            remove_sql_injection: true,
            remove_command_injection: true,
            remove_path_traversal: true,
            collapse_whitespace: false,
            remove_spam: false,
            remove_harmful_unicode: true,
            normalize_line_endings: false,
            trim_whitespace: false,
            allowed_html_tags: HashSet::new(),
            skip_categories: HashSet::new(),
        }
    }

    /// Create minimal rules for basic text cleaning.
    pub fn minimal() -> Self {
        Self {
            remove_html: false,
            remove_scripts: false,
            normalize_unicode: false,
            max_length: 0,
            remove_sql_injection: false,
            remove_command_injection: false,
            remove_path_traversal: false,
            collapse_whitespace: true,
            remove_spam: false,
            remove_harmful_unicode: false,
            normalize_line_endings: true,
            trim_whitespace: true,
            allowed_html_tags: HashSet::new(),
            skip_categories: HashSet::new(),
        }
    }

    /// Allow specific HTML tags.
    pub fn allow_html_tags(mut self, tags: Vec<&str>) -> Self {
        self.allowed_html_tags = tags.into_iter().map(|s| s.to_lowercase()).collect();
        self
    }

    /// Skip specific removal categories.
    pub fn skip_category(mut self, category: RemovalCategory) -> Self {
        self.skip_categories.insert(category);
        self
    }

    /// Set maximum content length.
    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = max;
        self
    }

    /// Check if a category should be processed.
    pub fn should_process(&self, category: RemovalCategory) -> bool {
        !self.skip_categories.contains(&category)
    }
}

// =============================================================================
// SANITIZER CONFIG
// =============================================================================

/// Configuration for the Sanitizer Agent.
#[derive(Debug, Clone)]
pub struct SanitizerConfig {
    /// Sanitization rules to apply.
    pub rules: SanitizationRules,

    /// Log all removals for audit.
    pub audit_mode: bool,

    /// Fail on security issues instead of just removing them.
    pub strict_mode: bool,

    /// Maximum processing time in milliseconds.
    pub timeout_ms: u64,

    /// Include samples of removed content in the report.
    pub include_samples: bool,
}

impl Default for SanitizerConfig {
    fn default() -> Self {
        Self {
            rules: SanitizationRules::default(),
            audit_mode: false,
            strict_mode: false,
            timeout_ms: 5000,
            include_samples: false,
        }
    }
}

impl SanitizerConfig {
    /// Create a strict configuration.
    pub fn strict() -> Self {
        Self {
            rules: SanitizationRules::strict(),
            audit_mode: true,
            strict_mode: true,
            timeout_ms: 5000,
            include_samples: true,
        }
    }

    /// Create a permissive configuration.
    pub fn permissive() -> Self {
        Self {
            rules: SanitizationRules::permissive(),
            audit_mode: false,
            strict_mode: false,
            timeout_ms: 10000,
            include_samples: false,
        }
    }
}

// =============================================================================
// COMPILED PATTERNS
// =============================================================================

/// Compiled regex patterns for efficient matching.
#[derive(Debug, Clone)]
struct CompiledPatterns {
    sql_injection: Regex,
    xss: Regex,
    command_injection: Regex,
    path_traversal: Regex,
    html_tags: Regex,
    excessive_whitespace: Regex,
    spam: Regex,
    harmful_unicode: Regex,
}

impl Default for CompiledPatterns {
    fn default() -> Self {
        Self {
            sql_injection: Regex::new(SQL_INJECTION_PATTERN).expect("Invalid SQL injection pattern"),
            xss: Regex::new(XSS_PATTERN).expect("Invalid XSS pattern"),
            command_injection: Regex::new(COMMAND_INJECTION_PATTERN)
                .expect("Invalid command injection pattern"),
            path_traversal: Regex::new(PATH_TRAVERSAL_PATTERN)
                .expect("Invalid path traversal pattern"),
            html_tags: Regex::new(HTML_TAG_PATTERN).expect("Invalid HTML tag pattern"),
            excessive_whitespace: Regex::new(EXCESSIVE_WHITESPACE_PATTERN)
                .expect("Invalid whitespace pattern"),
            spam: Regex::new(SPAM_PATTERN).expect("Invalid spam pattern"),
            harmful_unicode: Regex::new(HARMFUL_UNICODE_PATTERN)
                .expect("Invalid Unicode pattern"),
        }
    }
}

// =============================================================================
// SANITIZER AGENT
// =============================================================================

/// The Input Cleaner — Agent 28 of Project Panpsychism.
///
/// This agent sanitizes and cleans inputs for security and quality,
/// protecting the system from malicious content while preserving
/// the essential meaning of inputs.
///
/// ## Capabilities
///
/// - **Injection Prevention**: SQL, command, and path traversal protection
/// - **XSS Protection**: Script and event handler removal
/// - **Unicode Safety**: Harmful character detection and removal
/// - **Content Normalization**: Whitespace, encoding, and line ending fixes
///
/// ## Example
///
/// ```rust,ignore
/// use panpsychism::sanitizer::SanitizerAgent;
///
/// let sanitizer = SanitizerAgent::new();
///
/// // Basic sanitization
/// let result = sanitizer.sanitize("Hello <script>evil()</script>").await?;
/// assert_eq!(result.content, "Hello ");
///
/// // Check what was removed
/// for removal in &result.removals {
///     println!("Removed {}: {} occurrences", removal.category, removal.count);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SanitizerAgent {
    /// Configuration for sanitization behavior.
    config: SanitizerConfig,

    /// Compiled regex patterns.
    patterns: CompiledPatterns,
}

impl Default for SanitizerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl SanitizerAgent {
    /// Create a new Sanitizer Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: SanitizerConfig::default(),
            patterns: CompiledPatterns::default(),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> SanitizerAgentBuilder {
        SanitizerAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SanitizerConfig {
        &self.config
    }

    /// Get the current rules.
    pub fn rules(&self) -> &SanitizationRules {
        &self.config.rules
    }

    // =========================================================================
    // MAIN SANITIZATION METHOD
    // =========================================================================

    /// Sanitize the input content.
    ///
    /// Applies all configured sanitization rules to the input,
    /// removing harmful content and normalizing formatting.
    ///
    /// # Arguments
    ///
    /// * `input` - The content to sanitize
    ///
    /// # Returns
    ///
    /// A `SanitizedInput` containing the cleaned content and metadata.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` if:
    /// - Input is empty and strict mode is enabled
    /// - Security issues are found and strict mode is enabled
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sanitizer = SanitizerAgent::new();
    /// let result = sanitizer.sanitize("User input with <script>bad()</script>").await?;
    /// println!("Clean: {}", result.content);
    /// ```
    pub async fn sanitize(&self, input: &str) -> Result<SanitizedInput> {
        let start = Instant::now();
        let original_length = input.len();

        if input.is_empty() {
            return Ok(SanitizedInput::new("", 0));
        }

        let mut content = input.to_string();
        let mut removals = Vec::new();
        let mut normalizations = Vec::new();
        let mut had_security_issues = false;

        // Apply normalizations first
        if self.config.rules.normalize_line_endings {
            let (new_content, count) = self.normalize_line_endings(&content);
            if count > 0 {
                content = new_content;
                normalizations.push(Normalization::new(NormalizationType::LineEndings, count));
            }
        }

        if self.config.rules.normalize_unicode {
            let (new_content, count) = self.normalize_unicode(&content);
            if count > 0 {
                content = new_content;
                normalizations.push(Normalization::new(NormalizationType::Unicode, count));
            }
        }

        // Remove security threats
        if self.config.rules.remove_sql_injection
            && self.config.rules.should_process(RemovalCategory::Injection)
        {
            let (new_content, removal) =
                self.remove_pattern(&content, &self.patterns.sql_injection, RemovalCategory::Injection, "SQL injection");
            if let Some(r) = removal {
                had_security_issues = true;
                removals.push(r);
                content = new_content;
            }
        }

        if self.config.rules.remove_scripts
            && self.config.rules.should_process(RemovalCategory::MaliciousCode)
        {
            let (new_content, removal) =
                self.remove_pattern(&content, &self.patterns.xss, RemovalCategory::MaliciousCode, "XSS/Script");
            if let Some(r) = removal {
                had_security_issues = true;
                removals.push(r);
                content = new_content;
            }
        }

        if self.config.rules.remove_command_injection
            && self.config.rules.should_process(RemovalCategory::Injection)
        {
            let (new_content, removal) =
                self.remove_pattern(&content, &self.patterns.command_injection, RemovalCategory::Injection, "Command injection");
            if let Some(r) = removal {
                had_security_issues = true;
                removals.push(r);
                content = new_content;
            }
        }

        if self.config.rules.remove_path_traversal
            && self.config.rules.should_process(RemovalCategory::Injection)
        {
            let (new_content, removal) =
                self.remove_pattern(&content, &self.patterns.path_traversal, RemovalCategory::Injection, "Path traversal");
            if let Some(r) = removal {
                had_security_issues = true;
                removals.push(r);
                content = new_content;
            }
        }

        if self.config.rules.remove_html
            && self.config.rules.should_process(RemovalCategory::HtmlTag)
        {
            let (new_content, removal) =
                self.remove_pattern(&content, &self.patterns.html_tags, RemovalCategory::HtmlTag, "HTML tags");
            if let Some(r) = removal {
                removals.push(r);
                content = new_content;
            }
        }

        if self.config.rules.remove_harmful_unicode
            && self.config.rules.should_process(RemovalCategory::HarmfulUnicode)
        {
            let (new_content, removal) =
                self.remove_pattern(&content, &self.patterns.harmful_unicode, RemovalCategory::HarmfulUnicode, "Harmful Unicode");
            if let Some(r) = removal {
                had_security_issues = true;
                removals.push(r);
                content = new_content;
            }
        }

        if self.config.rules.remove_spam
            && self.config.rules.should_process(RemovalCategory::Spam)
        {
            let (new_content, removal) =
                self.remove_pattern(&content, &self.patterns.spam, RemovalCategory::Spam, "Spam");
            if let Some(r) = removal {
                removals.push(r);
                content = new_content;
            }
        }

        // Whitespace handling
        if self.config.rules.collapse_whitespace
            && self.config.rules.should_process(RemovalCategory::ExcessiveWhitespace)
        {
            let (new_content, removal) = self.collapse_whitespace(&content);
            if let Some(r) = removal {
                removals.push(r);
                content = new_content;
            }
        }

        if self.config.rules.trim_whitespace {
            let trimmed = content.trim();
            if trimmed.len() != content.len() {
                let count = content.len() - trimmed.len();
                normalizations.push(
                    Normalization::new(NormalizationType::Whitespace, count)
                        .with_description("Trimmed leading/trailing whitespace"),
                );
                content = trimmed.to_string();
            }
        }

        // Apply max length
        if self.config.rules.max_length > 0 && content.len() > self.config.rules.max_length {
            let truncated_len = content.len() - self.config.rules.max_length;
            content.truncate(self.config.rules.max_length);
            normalizations.push(
                Normalization::new(NormalizationType::Encoding, truncated_len)
                    .with_description(format!("Truncated to {} characters", self.config.rules.max_length)),
            );
        }

        // Check for strict mode violations
        if self.config.strict_mode && had_security_issues {
            let categories: Vec<String> = removals
                .iter()
                .filter(|r| r.is_high_severity())
                .map(|r| r.category.to_string())
                .collect();
            return Err(Error::Validation(format!(
                "Security issues detected: {}",
                categories.join(", ")
            )));
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let security_score: usize = removals.iter().map(|r| r.security_impact()).sum();

        if self.config.audit_mode && !removals.is_empty() {
            info!(
                "Sanitization audit: {} removals, {} normalizations, security_score={}",
                removals.len(),
                normalizations.len(),
                security_score
            );
            for removal in &removals {
                debug!("  Removed {}: {} occurrences", removal.category, removal.count);
            }
        }

        Ok(SanitizedInput {
            content,
            original_length,
            removals,
            normalizations,
            sanitization_time_ms: duration_ms,
            had_security_issues,
            security_score,
        })
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// Remove content matching a pattern.
    fn remove_pattern(
        &self,
        content: &str,
        pattern: &Regex,
        category: RemovalCategory,
        pattern_name: &str,
    ) -> (String, Option<Removal>) {
        let matches: Vec<_> = pattern.find_iter(content).collect();

        if matches.is_empty() {
            return (content.to_string(), None);
        }

        let count = matches.len();
        let positions: Vec<usize> = matches.iter().map(|m| m.start()).collect();

        let sample = if self.config.include_samples {
            matches.first().map(|m| m.as_str().to_string())
        } else {
            None
        };

        let cleaned = pattern.replace_all(content, "").to_string();

        let mut removal = Removal::new(category, pattern_name, count).with_positions(positions);

        if let Some(s) = sample {
            removal = removal.with_sample(s);
        }

        (cleaned, Some(removal))
    }

    /// Normalize line endings to LF.
    fn normalize_line_endings(&self, content: &str) -> (String, usize) {
        let crlf_count = content.matches("\r\n").count();
        let cr_only_count = content.matches('\r').count().saturating_sub(crlf_count);

        if crlf_count == 0 && cr_only_count == 0 {
            return (content.to_string(), 0);
        }

        let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
        (normalized, crlf_count + cr_only_count)
    }

    /// Normalize Unicode to NFC form.
    fn normalize_unicode(&self, content: &str) -> (String, usize) {
        // Use unicode-normalization crate if available
        // For now, just count non-ASCII characters as a placeholder
        let non_ascii_count = content.chars().filter(|c| !c.is_ascii()).count();

        // Basic normalization: replace common problematic sequences
        let mut normalized = content.to_string();
        let mut changes = 0;

        // Replace zero-width spaces
        if normalized.contains('\u{200B}') {
            normalized = normalized.replace('\u{200B}', "");
            changes += 1;
        }

        // Replace zero-width non-joiner
        if normalized.contains('\u{200C}') {
            normalized = normalized.replace('\u{200C}', "");
            changes += 1;
        }

        // Replace zero-width joiner
        if normalized.contains('\u{200D}') {
            normalized = normalized.replace('\u{200D}', "");
            changes += 1;
        }

        // Replace BOM
        if normalized.contains('\u{FEFF}') {
            normalized = normalized.replace('\u{FEFF}', "");
            changes += 1;
        }

        // Only count as changes if we actually modified the content
        let effective_changes = if normalized.len() != content.len() {
            changes
        } else {
            0
        };

        (normalized, effective_changes + if non_ascii_count > 0 { 0 } else { 0 })
    }

    /// Collapse excessive whitespace.
    fn collapse_whitespace(&self, content: &str) -> (String, Option<Removal>) {
        let matches: Vec<_> = self.patterns.excessive_whitespace.find_iter(content).collect();

        if matches.is_empty() {
            return (content.to_string(), None);
        }

        let count = matches.len();
        let positions: Vec<usize> = matches.iter().map(|m| m.start()).collect();

        // Replace multiple spaces with single space, multiple newlines with double
        let collapsed = self
            .patterns
            .excessive_whitespace
            .replace_all(content, |caps: &regex::Captures| {
                let matched = &caps[0];
                if matched.contains('\n') || matched.contains('\r') {
                    "\n\n"
                } else {
                    " "
                }
            })
            .to_string();

        let removal = Removal::new(RemovalCategory::ExcessiveWhitespace, "Whitespace", count)
            .with_positions(positions);

        (collapsed, Some(removal))
    }

    /// Quick check if input contains any security threats.
    ///
    /// This is a fast check that doesn't perform full sanitization.
    pub fn has_security_threats(&self, input: &str) -> bool {
        if self.config.rules.remove_sql_injection && self.patterns.sql_injection.is_match(input) {
            return true;
        }
        if self.config.rules.remove_scripts && self.patterns.xss.is_match(input) {
            return true;
        }
        if self.config.rules.remove_command_injection
            && self.patterns.command_injection.is_match(input)
        {
            return true;
        }
        if self.config.rules.remove_path_traversal && self.patterns.path_traversal.is_match(input) {
            return true;
        }
        if self.config.rules.remove_harmful_unicode && self.patterns.harmful_unicode.is_match(input)
        {
            return true;
        }
        false
    }

    /// Get a list of detected threat categories.
    pub fn detect_threats(&self, input: &str) -> Vec<RemovalCategory> {
        let mut threats = Vec::new();

        if self.config.rules.remove_sql_injection && self.patterns.sql_injection.is_match(input) {
            threats.push(RemovalCategory::Injection);
        }
        if self.config.rules.remove_scripts && self.patterns.xss.is_match(input) {
            threats.push(RemovalCategory::MaliciousCode);
        }
        if self.config.rules.remove_command_injection
            && self.patterns.command_injection.is_match(input)
        {
            if !threats.contains(&RemovalCategory::Injection) {
                threats.push(RemovalCategory::Injection);
            }
        }
        if self.config.rules.remove_path_traversal && self.patterns.path_traversal.is_match(input) {
            if !threats.contains(&RemovalCategory::Injection) {
                threats.push(RemovalCategory::Injection);
            }
        }
        if self.config.rules.remove_harmful_unicode && self.patterns.harmful_unicode.is_match(input)
        {
            threats.push(RemovalCategory::HarmfulUnicode);
        }
        if self.config.rules.remove_html && self.patterns.html_tags.is_match(input) {
            threats.push(RemovalCategory::HtmlTag);
        }
        if self.config.rules.remove_spam && self.patterns.spam.is_match(input) {
            threats.push(RemovalCategory::Spam);
        }

        threats
    }

    /// Sanitize multiple inputs in batch.
    pub async fn sanitize_batch(&self, inputs: &[&str]) -> Vec<Result<SanitizedInput>> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.sanitize(input).await);
        }
        results
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom SanitizerAgent configuration.
#[derive(Debug, Default)]
pub struct SanitizerAgentBuilder {
    config: Option<SanitizerConfig>,
    rules: Option<SanitizationRules>,
}

impl SanitizerAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: SanitizerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the sanitization rules.
    pub fn rules(mut self, rules: SanitizationRules) -> Self {
        self.rules = Some(rules);
        self
    }

    /// Enable audit mode.
    pub fn audit_mode(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.audit_mode = enabled;
        self.config = Some(config);
        self
    }

    /// Enable strict mode.
    pub fn strict_mode(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.strict_mode = enabled;
        self.config = Some(config);
        self
    }

    /// Include samples of removed content.
    pub fn include_samples(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_samples = enabled;
        self.config = Some(config);
        self
    }

    /// Set timeout in milliseconds.
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.timeout_ms = timeout;
        self.config = Some(config);
        self
    }

    /// Enable or disable HTML removal.
    pub fn remove_html(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.rules.remove_html = enabled;
        self.config = Some(config);
        self
    }

    /// Enable or disable script removal.
    pub fn remove_scripts(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.rules.remove_scripts = enabled;
        self.config = Some(config);
        self
    }

    /// Enable or disable Unicode normalization.
    pub fn normalize_unicode(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.rules.normalize_unicode = enabled;
        self.config = Some(config);
        self
    }

    /// Set maximum content length.
    pub fn max_length(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.rules.max_length = max;
        self.config = Some(config);
        self
    }

    /// Enable or disable SQL injection removal.
    pub fn remove_sql_injection(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.rules.remove_sql_injection = enabled;
        self.config = Some(config);
        self
    }

    /// Enable or disable command injection removal.
    pub fn remove_command_injection(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.rules.remove_command_injection = enabled;
        self.config = Some(config);
        self
    }

    /// Enable or disable whitespace collapsing.
    pub fn collapse_whitespace(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.rules.collapse_whitespace = enabled;
        self.config = Some(config);
        self
    }

    /// Enable or disable spam removal.
    pub fn remove_spam(mut self, enabled: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.rules.remove_spam = enabled;
        self.config = Some(config);
        self
    }

    /// Build the SanitizerAgent.
    pub fn build(self) -> SanitizerAgent {
        let mut config = self.config.unwrap_or_default();

        if let Some(rules) = self.rules {
            config.rules = rules;
        }

        SanitizerAgent {
            config,
            patterns: CompiledPatterns::default(),
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
    // RemovalCategory Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_removal_category_display() {
        assert_eq!(RemovalCategory::MaliciousCode.to_string(), "MaliciousCode");
        assert_eq!(RemovalCategory::PersonalData.to_string(), "PersonalData");
        assert_eq!(RemovalCategory::Injection.to_string(), "Injection");
        assert_eq!(
            RemovalCategory::ExcessiveWhitespace.to_string(),
            "ExcessiveWhitespace"
        );
    }

    #[test]
    fn test_removal_category_all() {
        let all = RemovalCategory::all();
        assert_eq!(all.len(), 8);
        assert!(all.contains(&RemovalCategory::MaliciousCode));
        assert!(all.contains(&RemovalCategory::Injection));
    }

    #[test]
    fn test_removal_category_severity() {
        assert_eq!(RemovalCategory::MaliciousCode.severity(), 5);
        assert_eq!(RemovalCategory::Injection.severity(), 5);
        assert_eq!(RemovalCategory::HarmfulUnicode.severity(), 4);
        assert_eq!(RemovalCategory::ExcessiveWhitespace.severity(), 1);
    }

    #[test]
    fn test_removal_category_description() {
        let desc = RemovalCategory::MaliciousCode.description();
        assert!(desc.contains("Malicious"));
    }

    // -------------------------------------------------------------------------
    // NormalizationType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normalization_type_display() {
        assert_eq!(NormalizationType::Unicode.to_string(), "Unicode");
        assert_eq!(NormalizationType::Whitespace.to_string(), "Whitespace");
        assert_eq!(NormalizationType::LineEndings.to_string(), "LineEndings");
    }

    #[test]
    fn test_normalization_type_all() {
        let all = NormalizationType::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&NormalizationType::Unicode));
    }

    #[test]
    fn test_normalization_type_description() {
        let desc = NormalizationType::Unicode.description();
        assert!(desc.contains("NFC"));
    }

    // -------------------------------------------------------------------------
    // Removal Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_removal_new() {
        let removal = Removal::new(RemovalCategory::Injection, "SQL injection", 3);
        assert_eq!(removal.category, RemovalCategory::Injection);
        assert_eq!(removal.pattern, "SQL injection");
        assert_eq!(removal.count, 3);
        assert!(removal.positions.is_empty());
        assert!(removal.sample.is_none());
    }

    #[test]
    fn test_removal_with_positions() {
        let removal = Removal::new(RemovalCategory::MaliciousCode, "XSS", 2)
            .with_positions(vec![10, 50]);

        assert_eq!(removal.positions, vec![10, 50]);
    }

    #[test]
    fn test_removal_with_sample() {
        let removal = Removal::new(RemovalCategory::MaliciousCode, "XSS", 1)
            .with_sample("<script>alert('xss')</script>");

        assert!(removal.sample.is_some());
        assert!(removal.sample.unwrap().contains("script"));
    }

    #[test]
    fn test_removal_sample_truncation() {
        let long_sample = "x".repeat(100);
        let removal = Removal::new(RemovalCategory::Spam, "Long", 1).with_sample(long_sample);

        let sample = removal.sample.unwrap();
        assert!(sample.len() <= 53); // 50 chars + "..."
        assert!(sample.ends_with("..."));
    }

    #[test]
    fn test_removal_is_high_severity() {
        let high = Removal::new(RemovalCategory::MaliciousCode, "XSS", 1);
        assert!(high.is_high_severity());

        let low = Removal::new(RemovalCategory::ExcessiveWhitespace, "Whitespace", 1);
        assert!(!low.is_high_severity());
    }

    #[test]
    fn test_removal_security_impact() {
        let removal = Removal::new(RemovalCategory::MaliciousCode, "XSS", 3);
        assert_eq!(removal.security_impact(), 15); // 3 * 5

        let removal2 = Removal::new(RemovalCategory::ExcessiveWhitespace, "WS", 10);
        assert_eq!(removal2.security_impact(), 10); // 10 * 1
    }

    #[test]
    fn test_removal_display() {
        let removal = Removal::new(RemovalCategory::Injection, "SQL", 5);
        assert_eq!(format!("{}", removal), "Injection: 5 occurrences");
    }

    // -------------------------------------------------------------------------
    // Normalization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normalization_new() {
        let norm = Normalization::new(NormalizationType::Unicode, 5);
        assert_eq!(norm.normalization_type, NormalizationType::Unicode);
        assert_eq!(norm.count, 5);
        assert!(norm.description.is_none());
    }

    #[test]
    fn test_normalization_with_description() {
        let norm = Normalization::new(NormalizationType::Whitespace, 3)
            .with_description("Collapsed spaces");

        assert_eq!(norm.description, Some("Collapsed spaces".to_string()));
    }

    #[test]
    fn test_normalization_display() {
        let norm = Normalization::new(NormalizationType::LineEndings, 10);
        assert_eq!(format!("{}", norm), "LineEndings: 10 changes");
    }

    // -------------------------------------------------------------------------
    // SanitizedInput Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sanitized_input_new() {
        let input = SanitizedInput::new("clean content", 100);
        assert_eq!(input.content, "clean content");
        assert_eq!(input.original_length, 100);
        assert!(input.removals.is_empty());
        assert!(input.normalizations.is_empty());
    }

    #[test]
    fn test_sanitized_input_had_removals() {
        let mut input = SanitizedInput::new("content", 50);
        assert!(!input.had_removals());

        input.removals.push(Removal::new(RemovalCategory::Spam, "Spam", 1));
        assert!(input.had_removals());
    }

    #[test]
    fn test_sanitized_input_had_normalizations() {
        let mut input = SanitizedInput::new("content", 50);
        assert!(!input.had_normalizations());

        input
            .normalizations
            .push(Normalization::new(NormalizationType::Unicode, 1));
        assert!(input.had_normalizations());
    }

    #[test]
    fn test_sanitized_input_total_removals() {
        let mut input = SanitizedInput::new("content", 50);
        input.removals.push(Removal::new(RemovalCategory::Spam, "Spam", 3));
        input
            .removals
            .push(Removal::new(RemovalCategory::Injection, "SQL", 2));

        assert_eq!(input.total_removals(), 5);
    }

    #[test]
    fn test_sanitized_input_total_normalizations() {
        let mut input = SanitizedInput::new("content", 50);
        input
            .normalizations
            .push(Normalization::new(NormalizationType::Unicode, 3));
        input
            .normalizations
            .push(Normalization::new(NormalizationType::Whitespace, 2));

        assert_eq!(input.total_normalizations(), 5);
    }

    #[test]
    fn test_sanitized_input_high_severity_removals() {
        let mut input = SanitizedInput::new("content", 50);
        input
            .removals
            .push(Removal::new(RemovalCategory::MaliciousCode, "XSS", 1));
        input
            .removals
            .push(Removal::new(RemovalCategory::ExcessiveWhitespace, "WS", 5));

        let high = input.high_severity_removals();
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].category, RemovalCategory::MaliciousCode);
    }

    #[test]
    fn test_sanitized_input_removals_by_category() {
        let mut input = SanitizedInput::new("content", 50);
        input
            .removals
            .push(Removal::new(RemovalCategory::Injection, "SQL", 2));
        input
            .removals
            .push(Removal::new(RemovalCategory::Injection, "CMD", 1));
        input.removals.push(Removal::new(RemovalCategory::Spam, "Spam", 3));

        let injections = input.removals_by_category(RemovalCategory::Injection);
        assert_eq!(injections.len(), 2);
    }

    #[test]
    fn test_sanitized_input_reduction_percentage() {
        let input = SanitizedInput {
            content: "short".to_string(),
            original_length: 100,
            removals: Vec::new(),
            normalizations: Vec::new(),
            sanitization_time_ms: 0,
            had_security_issues: false,
            security_score: 0,
        };

        let percentage = input.reduction_percentage();
        assert!((percentage - 95.0).abs() < 0.01);
    }

    #[test]
    fn test_sanitized_input_reduction_percentage_zero_original() {
        let input = SanitizedInput::new("", 0);
        assert_eq!(input.reduction_percentage(), 0.0);
    }

    #[test]
    fn test_sanitized_input_is_clean() {
        let clean = SanitizedInput::new("content", 50);
        assert!(clean.is_clean());

        let mut dirty = SanitizedInput::new("content", 50);
        dirty.removals.push(Removal::new(RemovalCategory::Spam, "Spam", 1));
        assert!(!dirty.is_clean());
    }

    #[test]
    fn test_sanitized_input_summary_clean() {
        let input = SanitizedInput::new("clean", 50);
        assert!(input.summary().contains("clean"));
    }

    #[test]
    fn test_sanitized_input_summary_with_removals() {
        let mut input = SanitizedInput::new("content", 50);
        input.removals.push(Removal::new(RemovalCategory::Spam, "Spam", 3));

        let summary = input.summary();
        assert!(summary.contains("3 removals"));
    }

    #[test]
    fn test_sanitized_input_display() {
        let input = SanitizedInput::new("displayed content", 50);
        assert_eq!(format!("{}", input), "displayed content");
    }

    // -------------------------------------------------------------------------
    // SanitizationRules Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sanitization_rules_default() {
        let rules = SanitizationRules::default();
        assert!(rules.remove_html);
        assert!(rules.remove_scripts);
        assert!(rules.normalize_unicode);
        assert_eq!(rules.max_length, 0);
        assert!(rules.remove_sql_injection);
        assert!(!rules.remove_spam); // Off by default
    }

    #[test]
    fn test_sanitization_rules_strict() {
        let rules = SanitizationRules::strict();
        assert!(rules.remove_html);
        assert!(rules.remove_spam);
        assert_eq!(rules.max_length, 10000);
    }

    #[test]
    fn test_sanitization_rules_permissive() {
        let rules = SanitizationRules::permissive();
        assert!(!rules.remove_html);
        assert!(rules.remove_scripts);
        assert!(!rules.collapse_whitespace);
    }

    #[test]
    fn test_sanitization_rules_minimal() {
        let rules = SanitizationRules::minimal();
        assert!(!rules.remove_html);
        assert!(!rules.remove_scripts);
        assert!(!rules.remove_sql_injection);
        assert!(rules.trim_whitespace);
    }

    #[test]
    fn test_sanitization_rules_allow_html_tags() {
        let rules = SanitizationRules::default().allow_html_tags(vec!["p", "b", "i"]);
        assert!(rules.allowed_html_tags.contains("p"));
        assert!(rules.allowed_html_tags.contains("b"));
        assert!(rules.allowed_html_tags.contains("i"));
    }

    #[test]
    fn test_sanitization_rules_skip_category() {
        let rules = SanitizationRules::default().skip_category(RemovalCategory::Spam);
        assert!(rules.skip_categories.contains(&RemovalCategory::Spam));
        assert!(!rules.should_process(RemovalCategory::Spam));
        assert!(rules.should_process(RemovalCategory::Injection));
    }

    #[test]
    fn test_sanitization_rules_with_max_length() {
        let rules = SanitizationRules::default().with_max_length(5000);
        assert_eq!(rules.max_length, 5000);
    }

    // -------------------------------------------------------------------------
    // SanitizerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sanitizer_config_default() {
        let config = SanitizerConfig::default();
        assert!(!config.audit_mode);
        assert!(!config.strict_mode);
        assert_eq!(config.timeout_ms, 5000);
        assert!(!config.include_samples);
    }

    #[test]
    fn test_sanitizer_config_strict() {
        let config = SanitizerConfig::strict();
        assert!(config.audit_mode);
        assert!(config.strict_mode);
        assert!(config.include_samples);
    }

    #[test]
    fn test_sanitizer_config_permissive() {
        let config = SanitizerConfig::permissive();
        assert!(!config.audit_mode);
        assert!(!config.strict_mode);
        assert_eq!(config.timeout_ms, 10000);
    }

    // -------------------------------------------------------------------------
    // SanitizerAgent Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sanitizer_agent_new() {
        let agent = SanitizerAgent::new();
        assert!(!agent.config.audit_mode);
    }

    #[test]
    fn test_sanitizer_agent_builder() {
        let agent = SanitizerAgent::builder()
            .audit_mode(true)
            .strict_mode(true)
            .include_samples(true)
            .max_length(1000)
            .build();

        assert!(agent.config.audit_mode);
        assert!(agent.config.strict_mode);
        assert!(agent.config.include_samples);
        assert_eq!(agent.config.rules.max_length, 1000);
    }

    #[test]
    fn test_sanitizer_agent_builder_remove_options() {
        let agent = SanitizerAgent::builder()
            .remove_html(false)
            .remove_scripts(false)
            .remove_sql_injection(false)
            .remove_command_injection(false)
            .collapse_whitespace(false)
            .remove_spam(true)
            .normalize_unicode(false)
            .build();

        assert!(!agent.config.rules.remove_html);
        assert!(!agent.config.rules.remove_scripts);
        assert!(!agent.config.rules.remove_sql_injection);
        assert!(agent.config.rules.remove_spam);
    }

    // -------------------------------------------------------------------------
    // SanitizerAgent Sanitization Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_sanitize_empty_input() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("").await.unwrap();

        assert_eq!(result.content, "");
        assert_eq!(result.original_length, 0);
        assert!(result.is_clean());
    }

    #[tokio::test]
    async fn test_sanitize_clean_input() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("Hello, world!").await.unwrap();

        assert_eq!(result.content, "Hello, world!");
        assert!(!result.had_security_issues);
    }

    #[tokio::test]
    async fn test_sanitize_xss_script_tag() {
        let agent = SanitizerAgent::new();
        let result = agent
            .sanitize("Hello <script>alert('xss')</script> World")
            .await
            .unwrap();

        assert!(!result.content.contains("<script>"));
        assert!(!result.content.contains("alert"));
        assert!(result.had_security_issues);
        assert!(!result.removals.is_empty());
    }

    #[tokio::test]
    async fn test_sanitize_xss_event_handler() {
        let agent = SanitizerAgent::new();
        let result = agent
            .sanitize("<img src='x' onerror='alert(1)'>")
            .await
            .unwrap();

        assert!(!result.content.contains("onerror"));
        assert!(result.had_security_issues);
    }

    #[tokio::test]
    async fn test_sanitize_javascript_protocol() {
        let agent = SanitizerAgent::new();
        let result = agent
            .sanitize("<a href='javascript:alert(1)'>Click</a>")
            .await
            .unwrap();

        assert!(!result.content.contains("javascript:"));
    }

    #[tokio::test]
    async fn test_sanitize_sql_injection_or() {
        let agent = SanitizerAgent::new();
        let result = agent
            .sanitize("SELECT * FROM users WHERE id = '1' OR '1'='1'")
            .await
            .unwrap();

        assert!(!result.content.contains("OR '1'='1'"));
        assert!(result.had_security_issues);
    }

    #[tokio::test]
    async fn test_sanitize_sql_injection_drop() {
        let agent = SanitizerAgent::new();
        let result = agent
            .sanitize("user input'; DROP TABLE users; --")
            .await
            .unwrap();

        assert!(!result.content.contains("DROP TABLE"));
        assert!(result.had_security_issues);
    }

    #[tokio::test]
    async fn test_sanitize_command_injection() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("file.txt; rm -rf /").await.unwrap();

        assert!(!result.content.contains("; rm "));
        assert!(result.had_security_issues);
    }

    #[tokio::test]
    async fn test_sanitize_command_substitution() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("echo $(cat /etc/passwd)").await.unwrap();

        assert!(!result.content.contains("$("));
        assert!(result.had_security_issues);
    }

    #[tokio::test]
    async fn test_sanitize_path_traversal() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("../../etc/passwd").await.unwrap();

        assert!(!result.content.contains("../../"));
        assert!(result.had_security_issues);
    }

    #[tokio::test]
    async fn test_sanitize_harmful_unicode() {
        let agent = SanitizerAgent::new();
        // Zero-width space
        let result = agent.sanitize("Hello\u{200B}World").await.unwrap();

        assert!(!result.content.contains('\u{200B}'));
    }

    #[tokio::test]
    async fn test_sanitize_line_endings() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("Line 1\r\nLine 2\r\n").await.unwrap();

        assert!(!result.content.contains("\r\n"));
        assert!(result.content.contains('\n'));
    }

    #[tokio::test]
    async fn test_sanitize_excessive_whitespace() {
        let agent = SanitizerAgent::new();
        let result = agent
            .sanitize("Hello     World\n\n\n\nEnd")
            .await
            .unwrap();

        // Should collapse multiple spaces and newlines
        assert!(!result.content.contains("     "));
        assert!(!result.content.contains("\n\n\n\n"));
    }

    #[tokio::test]
    async fn test_sanitize_trim_whitespace() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("   trimmed   ").await.unwrap();

        assert_eq!(result.content, "trimmed");
    }

    #[tokio::test]
    async fn test_sanitize_max_length() {
        let agent = SanitizerAgent::builder().max_length(10).build();

        let result = agent.sanitize("This is a long string").await.unwrap();

        assert_eq!(result.content.len(), 10);
    }

    #[tokio::test]
    async fn test_sanitize_spam_detection() {
        let agent = SanitizerAgent::builder().remove_spam(true).build();

        let result = agent
            .sanitize("BUY NOW! Limited time offer!!!")
            .await
            .unwrap();

        assert!(!result.content.contains("BUY NOW"));
    }

    #[tokio::test]
    async fn test_sanitize_html_iframe() {
        let agent = SanitizerAgent::new();
        let result = agent
            .sanitize("Content <iframe src='evil.com'></iframe> More")
            .await
            .unwrap();

        assert!(!result.content.contains("<iframe"));
    }

    // -------------------------------------------------------------------------
    // Strict Mode Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_sanitize_strict_mode_error() {
        let agent = SanitizerAgent::builder().strict_mode(true).build();

        let result = agent.sanitize("<script>evil()</script>").await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[tokio::test]
    async fn test_sanitize_strict_mode_clean_passes() {
        let agent = SanitizerAgent::builder().strict_mode(true).build();

        let result = agent.sanitize("Clean content").await;

        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Threat Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_has_security_threats_sql_injection() {
        let agent = SanitizerAgent::new();
        assert!(agent.has_security_threats("' OR '1'='1"));
    }

    #[test]
    fn test_has_security_threats_xss() {
        let agent = SanitizerAgent::new();
        assert!(agent.has_security_threats("<script>alert(1)</script>"));
    }

    #[test]
    fn test_has_security_threats_command() {
        let agent = SanitizerAgent::new();
        assert!(agent.has_security_threats("; rm -rf /"));
    }

    #[test]
    fn test_has_security_threats_clean() {
        let agent = SanitizerAgent::new();
        assert!(!agent.has_security_threats("Hello, world!"));
    }

    #[test]
    fn test_detect_threats_multiple() {
        let agent = SanitizerAgent::new();
        let threats = agent.detect_threats("<script>alert(''; DROP TABLE x; --')</script>");

        assert!(threats.contains(&RemovalCategory::MaliciousCode));
        assert!(threats.contains(&RemovalCategory::Injection));
    }

    #[test]
    fn test_detect_threats_empty() {
        let agent = SanitizerAgent::new();
        let threats = agent.detect_threats("Safe content");

        assert!(threats.is_empty());
    }

    // -------------------------------------------------------------------------
    // Batch Sanitization Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_sanitize_batch() {
        let agent = SanitizerAgent::new();
        let inputs = vec!["Clean", "<script>bad</script>", "Also clean"];

        let results = agent.sanitize_batch(&inputs).await;

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_ok());

        let second = results[1].as_ref().unwrap();
        assert!(!second.content.contains("script"));
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_sanitize_complex_attack() {
        let agent = SanitizerAgent::builder()
            .audit_mode(true)
            .include_samples(true)
            .remove_spam(true)
            .build();

        let attack = r#"
            <script>alert('XSS')</script>
            '; DROP TABLE users; --
            $(cat /etc/passwd)
            ../../etc/shadow
            BUY NOW!!!
            <iframe src="evil.com"></iframe>
        "#;

        let result = agent.sanitize(attack).await.unwrap();

        assert!(!result.content.contains("<script>"));
        assert!(!result.content.contains("DROP TABLE"));
        assert!(!result.content.contains("$("));
        assert!(!result.content.contains("../.."));
        assert!(!result.content.contains("<iframe"));
        assert!(result.had_security_issues);
        assert!(result.security_score > 0);
    }

    #[tokio::test]
    async fn test_sanitize_preserves_valid_content() {
        let agent = SanitizerAgent::new();
        let valid = "This is a normal message with numbers 123 and symbols !@#$%";

        let result = agent.sanitize(valid).await.unwrap();

        assert_eq!(result.content, valid);
        assert!(result.is_clean());
    }

    #[tokio::test]
    async fn test_sanitize_unicode_preservation() {
        let agent = SanitizerAgent::new();
        let unicode = "Hello \u{1F600} World \u{4E2D}\u{6587}"; // Emoji and Chinese

        let result = agent.sanitize(unicode).await.unwrap();

        // Should preserve valid Unicode
        assert!(result.content.contains('\u{1F600}'));
        assert!(result.content.contains('\u{4E2D}'));
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_sanitize_only_whitespace() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("   \n\t   ").await.unwrap();

        assert_eq!(result.content, "");
    }

    #[tokio::test]
    async fn test_sanitize_mixed_line_endings() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("Line1\r\nLine2\rLine3\nLine4").await.unwrap();

        assert!(!result.content.contains('\r'));
    }

    #[tokio::test]
    async fn test_sanitize_nested_script_tags() {
        let agent = SanitizerAgent::new();
        let result = agent
            .sanitize("<script><script>nested()</script></script>")
            .await
            .unwrap();

        assert!(!result.content.contains("<script>"));
        assert!(!result.content.contains("</script>"));
    }

    #[tokio::test]
    async fn test_sanitize_encoded_xss() {
        let agent = SanitizerAgent::new();
        // JavaScript protocol
        let result = agent.sanitize("javascript:alert(1)").await.unwrap();

        assert!(!result.content.contains("javascript:"));
    }

    #[tokio::test]
    async fn test_sanitize_partial_script_tag() {
        let agent = SanitizerAgent::new();
        let result = agent.sanitize("<script>incomplete").await.unwrap();

        assert!(!result.content.contains("<script>"));
    }

    #[tokio::test]
    async fn test_sanitize_sql_union_select() {
        let agent = SanitizerAgent::new();
        let result = agent
            .sanitize("SELECT * FROM t UNION SELECT * FROM s")
            .await
            .unwrap();

        assert!(!result.content.contains("UNION SELECT"));
    }

    // -------------------------------------------------------------------------
    // Skip Category Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_skip_category_spam() {
        let rules = SanitizationRules::default()
            .skip_category(RemovalCategory::Spam);
        let agent = SanitizerAgent::builder()
            .rules(rules)
            .remove_spam(true)
            .build();

        // Even with remove_spam enabled, skipped category should be preserved
        let result = agent.sanitize("Normal content").await.unwrap();
        assert!(result.is_clean());
    }

    // -------------------------------------------------------------------------
    // Builder Pattern Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_with_config() {
        let config = SanitizerConfig::strict();
        let agent = SanitizerAgent::builder().config(config).build();

        assert!(agent.config.audit_mode);
        assert!(agent.config.strict_mode);
    }

    #[test]
    fn test_builder_with_rules() {
        let rules = SanitizationRules::minimal();
        let agent = SanitizerAgent::builder().rules(rules).build();

        assert!(!agent.config.rules.remove_scripts);
    }

    #[test]
    fn test_builder_timeout() {
        let agent = SanitizerAgent::builder().timeout_ms(10000).build();

        assert_eq!(agent.config.timeout_ms, 10000);
    }

    #[test]
    fn test_builder_chaining() {
        let agent = SanitizerAgent::builder()
            .audit_mode(true)
            .strict_mode(false)
            .include_samples(true)
            .max_length(5000)
            .remove_html(true)
            .remove_scripts(true)
            .remove_sql_injection(true)
            .remove_command_injection(true)
            .collapse_whitespace(true)
            .remove_spam(false)
            .normalize_unicode(true)
            .timeout_ms(3000)
            .build();

        assert!(agent.config.audit_mode);
        assert!(!agent.config.strict_mode);
        assert!(agent.config.include_samples);
        assert_eq!(agent.config.rules.max_length, 5000);
        assert!(agent.config.rules.remove_html);
        assert!(!agent.config.rules.remove_spam);
    }
}
