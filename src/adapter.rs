//! Adapter Agent module for Project Panpsychism.
//!
//! The Shape-Shifter - "Every spell must flow between forms."
//!
//! This module implements the Adapter Agent, responsible for transforming
//! prompts between different formats and structures. Like a master alchemist
//! who transmutes elements, the Adapter understands the essence of content
//! and can reshape it while preserving its meaning.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, the Adapter Agent serves a crucial role:
//!
//! - **Raw Essence** (source format) enters the transformation chamber
//! - **The Shape-Shifter** (AdapterAgent) applies transmutation rules
//! - **Transformed Spell** (target format) emerges ready for use
//!
//! The Shape-Shifter can:
//! - Transform between data formats (JSON, YAML, XML, etc.)
//! - Apply bidirectional transformations with lossless conversion
//! - Validate output against format schemas
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//! - **RATIO**: Logical preservation of structure through transformation
//! - **LAETITIA**: Joy through seamless format compatibility
//! - **NATURA**: Natural alignment between different representations
//! - **CONATUS**: Drive to maintain essence through form changes
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::adapter::{AdapterAgent, TargetFormat};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let adapter = AdapterAgent::new();
//!
//!     // Transform JSON to YAML
//!     let json = r#"{"name": "test", "value": 42}"#;
//!     let adapted = adapter.adapt(json, TargetFormat::Yaml).await?;
//!
//!     println!("{}", adapted.content);
//!     Ok(())
//! }
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// TARGET FORMAT ENUM
// =============================================================================

/// The target formats the Shape-Shifter can transform content into.
///
/// Each format represents a different way of structuring content,
/// suitable for different purposes and consumers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetFormat {
    /// JSON - JavaScript Object Notation.
    ///
    /// Machine-readable, supports nested structures.
    /// Ideal for APIs, configuration, and data exchange.
    Json,

    /// YAML - Human-friendly data serialization.
    ///
    /// Similar to JSON but more readable for humans.
    /// Ideal for configuration files and readable data.
    Yaml,

    /// XML - Extensible Markup Language.
    ///
    /// Verbose but well-structured markup format.
    /// Ideal for enterprise systems and document exchange.
    Xml,

    /// Markdown - Universal documentation format.
    ///
    /// Rich text with headers, emphasis, lists, and code blocks.
    /// Ideal for documentation, READMEs, and human-readable output.
    Markdown,

    /// PlainText - Raw, unformatted text.
    ///
    /// No formatting, just the essence of the content.
    /// Ideal for simple output, logs, and terminal display.
    PlainText,

    /// HTML - HyperText Markup Language.
    ///
    /// Web-ready markup format.
    /// Ideal for web pages and rich content display.
    Html,

    /// LaTeX - Document preparation system.
    ///
    /// Professional typesetting format.
    /// Ideal for academic papers and technical documentation.
    Latex,

    /// Custom format with a format string template.
    ///
    /// User-defined format with custom structure.
    /// Template syntax: `{key}` for value substitution.
    Custom(String),
}

impl Default for TargetFormat {
    fn default() -> Self {
        Self::Json
    }
}

impl std::fmt::Display for TargetFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::Yaml => write!(f, "yaml"),
            Self::Xml => write!(f, "xml"),
            Self::Markdown => write!(f, "markdown"),
            Self::PlainText => write!(f, "plaintext"),
            Self::Html => write!(f, "html"),
            Self::Latex => write!(f, "latex"),
            Self::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

impl std::str::FromStr for TargetFormat {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let lower = s.to_lowercase();

        // Handle custom:template format
        if lower.starts_with("custom:") {
            let template = lower.strip_prefix("custom:").unwrap_or("").to_string();
            return Ok(Self::Custom(template));
        }

        match lower.as_str() {
            "json" => Ok(Self::Json),
            "yaml" | "yml" => Ok(Self::Yaml),
            "xml" => Ok(Self::Xml),
            "markdown" | "md" => Ok(Self::Markdown),
            "plaintext" | "plain" | "text" | "txt" => Ok(Self::PlainText),
            "html" | "htm" => Ok(Self::Html),
            "latex" | "tex" => Ok(Self::Latex),
            _ => Err(Error::Config(format!(
                "Unknown target format: '{}'. Valid formats: json, yaml, xml, markdown, \
                 plaintext, html, latex, custom:<template>",
                s
            ))),
        }
    }
}

impl TargetFormat {
    /// Get the file extension for this format.
    pub fn extension(&self) -> &str {
        match self {
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Xml => "xml",
            Self::Markdown => "md",
            Self::PlainText => "txt",
            Self::Html => "html",
            Self::Latex => "tex",
            Self::Custom(_) => "txt",
        }
    }

    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &str {
        match self {
            Self::Json => "application/json",
            Self::Yaml => "application/x-yaml",
            Self::Xml => "application/xml",
            Self::Markdown => "text/markdown",
            Self::PlainText => "text/plain",
            Self::Html => "text/html",
            Self::Latex => "application/x-latex",
            Self::Custom(_) => "text/plain",
        }
    }

    /// Get all standard formats (excluding Custom).
    pub fn all_standard() -> Vec<Self> {
        vec![
            Self::Json,
            Self::Yaml,
            Self::Xml,
            Self::Markdown,
            Self::PlainText,
            Self::Html,
            Self::Latex,
        ]
    }
}

// =============================================================================
// VALIDATION RESULT
// =============================================================================

/// The result of format validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the content is valid for the target format.
    pub is_valid: bool,

    /// Validation errors, if any.
    pub errors: Vec<String>,

    /// Validation warnings (non-fatal issues).
    pub warnings: Vec<String>,

    /// Confidence score for the validation (0.0 - 1.0).
    pub confidence: f64,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            confidence: 1.0,
        }
    }
}

impl ValidationResult {
    /// Create a successful validation result.
    pub fn success() -> Self {
        Self::default()
    }

    /// Create a failed validation result with errors.
    pub fn failure(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
            confidence: 0.0,
        }
    }

    /// Create a validation result with warnings.
    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        if !self.warnings.is_empty() && self.confidence > 0.8 {
            self.confidence = 0.8;
        }
        self
    }

    /// Add an error to the validation result.
    pub fn add_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
        self.is_valid = false;
        self.confidence = 0.0;
    }

    /// Add a warning to the validation result.
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
        if self.confidence > 0.8 {
            self.confidence = 0.8;
        }
    }

    /// Check if there are any warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Check if there are any errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

// =============================================================================
// TRANSFORMATION RULE
// =============================================================================

/// A rule for transforming content from one pattern to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRule {
    /// The source pattern to match (regex or literal).
    pub source_pattern: String,

    /// The target pattern to replace with.
    pub target_pattern: String,

    /// Priority of this rule (higher = applied first).
    pub priority: u32,

    /// Description of what this rule does.
    pub description: Option<String>,

    /// Whether this rule is bidirectional.
    pub bidirectional: bool,
}

impl TransformationRule {
    /// Create a new transformation rule.
    pub fn new(
        source_pattern: impl Into<String>,
        target_pattern: impl Into<String>,
    ) -> Self {
        Self {
            source_pattern: source_pattern.into(),
            target_pattern: target_pattern.into(),
            priority: 0,
            description: None,
            bidirectional: false,
        }
    }

    /// Set the priority for this rule.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the description for this rule.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Mark this rule as bidirectional.
    pub fn bidirectional(mut self) -> Self {
        self.bidirectional = true;
        self
    }

    /// Apply this rule to content.
    pub fn apply(&self, content: &str) -> String {
        if let Ok(re) = regex::Regex::new(&self.source_pattern) {
            re.replace_all(content, self.target_pattern.as_str()).to_string()
        } else {
            // Fallback to literal replacement
            content.replace(&self.source_pattern, &self.target_pattern)
        }
    }

    /// Apply the reverse of this rule (if bidirectional).
    pub fn apply_reverse(&self, content: &str) -> Option<String> {
        if !self.bidirectional {
            return None;
        }

        if let Ok(re) = regex::Regex::new(&self.target_pattern) {
            Some(re.replace_all(content, self.source_pattern.as_str()).to_string())
        } else {
            Some(content.replace(&self.target_pattern, &self.source_pattern))
        }
    }
}

// =============================================================================
// ADAPTER CONFIG
// =============================================================================

/// Configuration for the Adapter Agent.
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Preserve the structure of nested objects.
    pub preserve_structure: bool,

    /// Validate output against format schema.
    pub validate_output: bool,

    /// Include schema information in output.
    pub include_schema: bool,

    /// Pretty-print output for human readability.
    pub pretty_print: bool,

    /// Maximum nesting depth for structured formats.
    pub max_depth: usize,

    /// Timeout in seconds for transformation operations.
    pub timeout_secs: u64,

    /// Custom transformation rules.
    pub rules: Vec<TransformationRule>,

    /// Whether to enable bidirectional transformations.
    pub enable_bidirectional: bool,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            preserve_structure: true,
            validate_output: true,
            include_schema: false,
            pretty_print: true,
            max_depth: 10,
            timeout_secs: 30,
            rules: Vec::new(),
            enable_bidirectional: true,
        }
    }
}

impl AdapterConfig {
    /// Create a configuration optimized for machine consumption.
    pub fn machine() -> Self {
        Self {
            preserve_structure: true,
            validate_output: true,
            include_schema: true,
            pretty_print: false,
            max_depth: 20,
            timeout_secs: 60,
            rules: Vec::new(),
            enable_bidirectional: true,
        }
    }

    /// Create a configuration optimized for human readability.
    pub fn human() -> Self {
        Self {
            preserve_structure: true,
            validate_output: false,
            include_schema: false,
            pretty_print: true,
            max_depth: 5,
            timeout_secs: 30,
            rules: Vec::new(),
            enable_bidirectional: true,
        }
    }

    /// Add a transformation rule.
    pub fn add_rule(mut self, rule: TransformationRule) -> Self {
        self.rules.push(rule);
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
        self
    }
}

// =============================================================================
// ADAPTED PROMPT
// =============================================================================

/// The result of content adaptation.
///
/// Contains the transformed content along with metadata about
/// the transformation process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptedPrompt {
    /// The transformed content string.
    pub content: String,

    /// The target format applied.
    pub format: TargetFormat,

    /// Schema information (if available).
    pub schema: Option<String>,

    /// Validation result for the output.
    pub validation_result: ValidationResult,

    /// Original content before transformation.
    pub original: String,

    /// Source format detected from input.
    pub source_format: Option<TargetFormat>,

    /// Processing duration in milliseconds.
    pub duration_ms: u64,

    /// Transformation rules applied.
    pub rules_applied: Vec<String>,
}

impl AdaptedPrompt {
    /// Create a new adapted prompt.
    pub fn new(content: impl Into<String>, format: TargetFormat) -> Self {
        let content = content.into();
        Self {
            original: content.clone(),
            content,
            format,
            schema: None,
            validation_result: ValidationResult::default(),
            source_format: None,
            duration_ms: 0,
            rules_applied: Vec::new(),
        }
    }

    /// Check if the adaptation was successful.
    pub fn is_valid(&self) -> bool {
        self.validation_result.is_valid
    }

    /// Check if there were any warnings during adaptation.
    pub fn has_warnings(&self) -> bool {
        self.validation_result.has_warnings()
    }

    /// Get the size change from adaptation.
    pub fn size_delta(&self) -> isize {
        self.content.len() as isize - self.original.len() as isize
    }

    /// Check if the content was actually transformed.
    pub fn was_transformed(&self) -> bool {
        self.content != self.original
    }

    /// Set the source format.
    pub fn with_source_format(mut self, format: TargetFormat) -> Self {
        self.source_format = Some(format);
        self
    }

    /// Set the schema.
    pub fn with_schema(mut self, schema: impl Into<String>) -> Self {
        self.schema = Some(schema.into());
        self
    }
}

// =============================================================================
// ADAPTER AGENT
// =============================================================================

/// The Adapter Agent - The Shape-Shifter of the Sorcerer's Tower.
///
/// Responsible for all format transformations, bidirectional conversions,
/// and format validation. The Shape-Shifter ensures that content can flow
/// freely between different representations while preserving its essence.
///
/// # Example
///
/// ```rust,ignore
/// let adapter = AdapterAgent::new();
///
/// // Transform JSON to YAML
/// let result = adapter.adapt(r#"{"key": "value"}"#, TargetFormat::Yaml).await?;
///
/// // Detect format and transform
/// let result = adapter.auto_adapt(content, TargetFormat::Json).await?;
///
/// // Validate format
/// let is_valid = adapter.validate(content, &TargetFormat::Json);
/// ```
#[derive(Debug, Clone)]
pub struct AdapterAgent {
    /// Configuration for adaptation behavior.
    config: AdapterConfig,

    /// Format-specific transformation rules.
    format_rules: HashMap<String, Vec<TransformationRule>>,
}

impl Default for AdapterAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterAgent {
    /// Create a new Adapter Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: AdapterConfig::default(),
            format_rules: Self::default_format_rules(),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> AdapterAgentBuilder {
        AdapterAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &AdapterConfig {
        &self.config
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: AdapterConfig) -> Self {
        self.config = config;
        self
    }

    /// Get default format transformation rules.
    fn default_format_rules() -> HashMap<String, Vec<TransformationRule>> {
        let mut rules = HashMap::new();

        // JSON to XML rules
        rules.insert(
            "json_to_xml".to_string(),
            vec![
                TransformationRule::new(r#"\{|\}"#, "").with_priority(1),
            ],
        );

        // Markdown to HTML rules
        rules.insert(
            "markdown_to_html".to_string(),
            vec![
                TransformationRule::new(r#"^# (.+)$"#, "<h1>$1</h1>")
                    .with_priority(10)
                    .bidirectional(),
                TransformationRule::new(r#"^## (.+)$"#, "<h2>$1</h2>")
                    .with_priority(9)
                    .bidirectional(),
                TransformationRule::new(r#"^### (.+)$"#, "<h3>$1</h3>")
                    .with_priority(8)
                    .bidirectional(),
                TransformationRule::new(r#"\*\*(.+?)\*\*"#, "<strong>$1</strong>")
                    .with_priority(5)
                    .bidirectional(),
                TransformationRule::new(r#"\*(.+?)\*"#, "<em>$1</em>")
                    .with_priority(4)
                    .bidirectional(),
            ],
        );

        rules
    }

    // =========================================================================
    // MAIN ADAPTATION METHOD
    // =========================================================================

    /// Adapt content to the specified target format.
    ///
    /// This is the primary method of the Shape-Shifter, channeling magical
    /// energy to transform content between different representations.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to transform
    /// * `target` - The target format
    ///
    /// # Returns
    ///
    /// An `AdaptedPrompt` containing the transformed content and metadata.
    ///
    /// # Errors
    ///
    /// Returns `Error::Synthesis` if transformation fails.
    pub async fn adapt(&self, content: &str, target: TargetFormat) -> Result<AdaptedPrompt> {
        let start = std::time::Instant::now();

        if content.trim().is_empty() {
            return Err(Error::Synthesis("Cannot adapt empty content".to_string()));
        }

        // Detect source format
        let source_format = self.detect_format(content);

        // Transform content
        let transformed = self.transform(content, &source_format, &target)?;

        // Validate if configured
        let validation_result = if self.config.validate_output {
            self.validate(&transformed, &target)
        } else {
            ValidationResult::success()
        };

        // Generate schema if configured
        let schema = if self.config.include_schema {
            Some(self.generate_schema(&target))
        } else {
            None
        };

        Ok(AdaptedPrompt {
            original: content.to_string(),
            content: transformed,
            format: target,
            schema,
            validation_result,
            source_format: Some(source_format),
            duration_ms: start.elapsed().as_millis() as u64,
            rules_applied: Vec::new(),
        })
    }

    /// Adapt content with automatic format detection.
    pub async fn auto_adapt(&self, content: &str, target: TargetFormat) -> Result<AdaptedPrompt> {
        self.adapt(content, target).await
    }

    /// Perform bidirectional transformation (target to source).
    pub async fn reverse_adapt(
        &self,
        content: &str,
        source: TargetFormat,
        target: TargetFormat,
    ) -> Result<AdaptedPrompt> {
        if !self.config.enable_bidirectional {
            return Err(Error::Config(
                "Bidirectional transformations are disabled".to_string(),
            ));
        }

        // Transform from target format back to source format
        self.adapt(content, source).await.map(|mut result| {
            result.source_format = Some(target);
            result
        })
    }

    // =========================================================================
    // FORMAT DETECTION
    // =========================================================================

    /// Detect the format of the given content.
    pub fn detect_format(&self, content: &str) -> TargetFormat {
        let trimmed = content.trim();

        if trimmed.is_empty() {
            return TargetFormat::PlainText;
        }

        // Check for JSON
        if (trimmed.starts_with('{') && trimmed.ends_with('}'))
            || (trimmed.starts_with('[') && trimmed.ends_with(']'))
        {
            if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
                return TargetFormat::Json;
            }
        }

        // Check for HTML first (before XML since HTML is also tag-based)
        if trimmed.contains("<!DOCTYPE html")
            || trimmed.contains("<!doctype html")
            || trimmed.contains("<html")
            || self.looks_like_html(trimmed)
        {
            return TargetFormat::Html;
        }

        // Check for XML
        if trimmed.starts_with("<?xml") || (trimmed.starts_with('<') && trimmed.ends_with('>')) {
            if self.looks_like_xml(trimmed) {
                return TargetFormat::Xml;
            }
        }

        // Check for YAML
        if trimmed.starts_with("---") || self.looks_like_yaml(trimmed) {
            return TargetFormat::Yaml;
        }

        // Check for Markdown
        if self.looks_like_markdown(trimmed) {
            return TargetFormat::Markdown;
        }

        // Check for LaTeX
        if trimmed.contains("\\documentclass") || trimmed.contains("\\begin{") {
            return TargetFormat::Latex;
        }

        TargetFormat::PlainText
    }

    /// Check if content looks like YAML.
    fn looks_like_yaml(&self, content: &str) -> bool {
        // Match key: value pattern (value can be on same line or next lines)
        let key_pattern = regex::Regex::new(r"^\s*[\w-]+:\s*").ok();
        // Match array items
        let array_pattern = regex::Regex::new(r"^\s*-\s+").ok();

        if let (Some(key_re), Some(arr_re)) = (key_pattern, array_pattern) {
            let key_lines = content.lines().filter(|l| key_re.is_match(l)).count();
            let arr_lines = content.lines().filter(|l| arr_re.is_match(l)).count();
            let total_lines = content.lines().count();

            // If we have key:value patterns or a mix of keys and arrays, it's YAML
            if key_lines > 0 && (key_lines + arr_lines) >= total_lines / 2 {
                // Try to actually parse it as YAML
                if serde_yaml::from_str::<serde_yaml::Value>(content).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// Check if content looks like Markdown.
    fn looks_like_markdown(&self, content: &str) -> bool {
        let has_header = content.lines().any(|l| l.starts_with('#'));
        // List items in markdown are typically followed by text, not just numbers
        let has_list = content.lines().any(|l| {
            let trimmed = l.trim();
            (trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("+ "))
                && trimmed.len() > 3
                && !trimmed[2..].trim().parse::<f64>().is_ok()  // Not just a number (YAML array)
        });
        let has_emphasis = content.contains("**") || content.contains("__");
        let has_code = content.contains("```") || content.contains('`');
        let has_link = content.contains("](") || content.contains("[](");

        let features = [has_header, has_list, has_emphasis, has_code, has_link];
        // Need header OR at least 2 other features to be markdown
        features.iter().filter(|&&x| x).count() >= 2 || has_header
    }

    /// Check if content looks like HTML.
    fn looks_like_html(&self, content: &str) -> bool {
        // Common HTML tags
        let html_tags = [
            "<body", "<head", "<div", "<span", "<p>", "<a ", "<img", "<table",
            "<ul>", "<ol>", "<li>", "<h1", "<h2", "<h3", "<h4", "<h5", "<h6",
            "<form", "<input", "<button", "<script", "<style", "<link", "<meta",
            "<header", "<footer", "<nav", "<main", "<section", "<article",
        ];

        let lower = content.to_lowercase();
        html_tags.iter().any(|tag| lower.contains(tag))
    }

    /// Check if content looks like XML.
    fn looks_like_xml(&self, content: &str) -> bool {
        let tag_pattern = regex::Regex::new(r"<[a-zA-Z][^>]*>").ok();
        if let Some(re) = tag_pattern {
            let tag_count = re.find_iter(content).count();
            return tag_count >= 2;
        }
        false
    }

    // =========================================================================
    // TRANSFORMATION
    // =========================================================================

    /// Transform content from source format to target format.
    fn transform(
        &self,
        content: &str,
        source: &TargetFormat,
        target: &TargetFormat,
    ) -> Result<String> {
        // If same format, just return cleaned content
        if source == target {
            return Ok(content.to_string());
        }

        match (source, target) {
            // JSON conversions
            (TargetFormat::Json, TargetFormat::Yaml) => self.json_to_yaml(content),
            (TargetFormat::Json, TargetFormat::Xml) => self.json_to_xml(content),
            (TargetFormat::Json, TargetFormat::Markdown) => self.json_to_markdown(content),
            (TargetFormat::Json, TargetFormat::PlainText) => self.json_to_plaintext(content),
            (TargetFormat::Json, TargetFormat::Html) => self.json_to_html(content),
            (TargetFormat::Json, TargetFormat::Latex) => self.json_to_latex(content),

            // YAML conversions
            (TargetFormat::Yaml, TargetFormat::Json) => self.yaml_to_json(content),
            (TargetFormat::Yaml, TargetFormat::Xml) => {
                let json = self.yaml_to_json(content)?;
                self.json_to_xml(&json)
            }
            (TargetFormat::Yaml, TargetFormat::Markdown) => {
                let json = self.yaml_to_json(content)?;
                self.json_to_markdown(&json)
            }
            (TargetFormat::Yaml, TargetFormat::PlainText) => {
                let json = self.yaml_to_json(content)?;
                self.json_to_plaintext(&json)
            }
            (TargetFormat::Yaml, TargetFormat::Html) => {
                let json = self.yaml_to_json(content)?;
                self.json_to_html(&json)
            }
            (TargetFormat::Yaml, TargetFormat::Latex) => {
                let json = self.yaml_to_json(content)?;
                self.json_to_latex(&json)
            }

            // Markdown conversions
            (TargetFormat::Markdown, TargetFormat::Html) => self.markdown_to_html(content),
            (TargetFormat::Markdown, TargetFormat::PlainText) => self.markdown_to_plaintext(content),
            (TargetFormat::Markdown, TargetFormat::Latex) => self.markdown_to_latex(content),
            (TargetFormat::Markdown, TargetFormat::Json) => self.markdown_to_json(content),
            (TargetFormat::Markdown, TargetFormat::Yaml) => {
                let json = self.markdown_to_json(content)?;
                self.json_to_yaml(&json)
            }
            (TargetFormat::Markdown, TargetFormat::Xml) => {
                let json = self.markdown_to_json(content)?;
                self.json_to_xml(&json)
            }

            // HTML conversions
            (TargetFormat::Html, TargetFormat::Markdown) => self.html_to_markdown(content),
            (TargetFormat::Html, TargetFormat::PlainText) => self.html_to_plaintext(content),
            (TargetFormat::Html, TargetFormat::Latex) => self.html_to_latex(content),
            (TargetFormat::Html, TargetFormat::Json) => self.html_to_json(content),
            (TargetFormat::Html, TargetFormat::Yaml) => {
                let json = self.html_to_json(content)?;
                self.json_to_yaml(&json)
            }
            (TargetFormat::Html, TargetFormat::Xml) => {
                let json = self.html_to_json(content)?;
                self.json_to_xml(&json)
            }

            // XML conversions
            (TargetFormat::Xml, TargetFormat::Json) => self.xml_to_json(content),
            (TargetFormat::Xml, TargetFormat::Yaml) => {
                let json = self.xml_to_json(content)?;
                self.json_to_yaml(&json)
            }
            (TargetFormat::Xml, TargetFormat::Markdown) => {
                let json = self.xml_to_json(content)?;
                self.json_to_markdown(&json)
            }
            (TargetFormat::Xml, TargetFormat::PlainText) => self.xml_to_plaintext(content),
            (TargetFormat::Xml, TargetFormat::Html) => self.xml_to_html(content),
            (TargetFormat::Xml, TargetFormat::Latex) => {
                let json = self.xml_to_json(content)?;
                self.json_to_latex(&json)
            }

            // PlainText conversions
            (TargetFormat::PlainText, TargetFormat::Json) => self.plaintext_to_json(content),
            (TargetFormat::PlainText, TargetFormat::Yaml) => self.plaintext_to_yaml(content),
            (TargetFormat::PlainText, TargetFormat::Markdown) => self.plaintext_to_markdown(content),
            (TargetFormat::PlainText, TargetFormat::Html) => self.plaintext_to_html(content),
            (TargetFormat::PlainText, TargetFormat::Xml) => self.plaintext_to_xml(content),
            (TargetFormat::PlainText, TargetFormat::Latex) => self.plaintext_to_latex(content),

            // LaTeX conversions
            (TargetFormat::Latex, TargetFormat::PlainText) => self.latex_to_plaintext(content),
            (TargetFormat::Latex, TargetFormat::Markdown) => self.latex_to_markdown(content),
            (TargetFormat::Latex, TargetFormat::Html) => self.latex_to_html(content),
            (TargetFormat::Latex, TargetFormat::Json) => self.latex_to_json(content),
            (TargetFormat::Latex, TargetFormat::Yaml) => {
                let json = self.latex_to_json(content)?;
                self.json_to_yaml(&json)
            }
            (TargetFormat::Latex, TargetFormat::Xml) => {
                let json = self.latex_to_json(content)?;
                self.json_to_xml(&json)
            }

            // Custom format
            (_, TargetFormat::Custom(template)) => self.apply_custom_template(content, template),
            (TargetFormat::Custom(_), _) => {
                // From custom, treat as plaintext
                self.transform(content, &TargetFormat::PlainText, target)
            }

            // Same format (should be caught by early return, but needed for exhaustiveness)
            (TargetFormat::Json, TargetFormat::Json)
            | (TargetFormat::Yaml, TargetFormat::Yaml)
            | (TargetFormat::Xml, TargetFormat::Xml)
            | (TargetFormat::Markdown, TargetFormat::Markdown)
            | (TargetFormat::PlainText, TargetFormat::PlainText)
            | (TargetFormat::Html, TargetFormat::Html)
            | (TargetFormat::Latex, TargetFormat::Latex) => Ok(content.to_string()),
        }
    }

    // =========================================================================
    // JSON CONVERSIONS
    // =========================================================================

    fn json_to_yaml(&self, content: &str) -> Result<String> {
        let value: serde_json::Value =
            serde_json::from_str(content).map_err(|e| Error::Synthesis(e.to_string()))?;
        serde_yaml::to_string(&value).map_err(|e| Error::Synthesis(e.to_string()))
    }

    fn json_to_xml(&self, content: &str) -> Result<String> {
        let value: serde_json::Value =
            serde_json::from_str(content).map_err(|e| Error::Synthesis(e.to_string()))?;
        Ok(self.value_to_xml(&value, "root", 0))
    }

    fn json_to_markdown(&self, content: &str) -> Result<String> {
        let value: serde_json::Value =
            serde_json::from_str(content).map_err(|e| Error::Synthesis(e.to_string()))?;
        Ok(self.value_to_markdown(&value, 1))
    }

    fn json_to_plaintext(&self, content: &str) -> Result<String> {
        let value: serde_json::Value =
            serde_json::from_str(content).map_err(|e| Error::Synthesis(e.to_string()))?;
        Ok(self.value_to_plaintext(&value, ""))
    }

    fn json_to_html(&self, content: &str) -> Result<String> {
        let value: serde_json::Value =
            serde_json::from_str(content).map_err(|e| Error::Synthesis(e.to_string()))?;
        Ok(self.value_to_html(&value))
    }

    fn json_to_latex(&self, content: &str) -> Result<String> {
        let value: serde_json::Value =
            serde_json::from_str(content).map_err(|e| Error::Synthesis(e.to_string()))?;
        Ok(self.value_to_latex(&value, 1))
    }

    // =========================================================================
    // YAML CONVERSIONS
    // =========================================================================

    fn yaml_to_json(&self, content: &str) -> Result<String> {
        let value: serde_json::Value =
            serde_yaml::from_str(content).map_err(|e| Error::Synthesis(e.to_string()))?;
        if self.config.pretty_print {
            serde_json::to_string_pretty(&value).map_err(|e| Error::Synthesis(e.to_string()))
        } else {
            serde_json::to_string(&value).map_err(|e| Error::Synthesis(e.to_string()))
        }
    }

    // =========================================================================
    // MARKDOWN CONVERSIONS
    // =========================================================================

    fn markdown_to_html(&self, content: &str) -> Result<String> {
        let mut html = content.to_string();

        // Headers
        let h1_re = regex::Regex::new(r"(?m)^# (.+)$").unwrap();
        html = h1_re.replace_all(&html, "<h1>$1</h1>").to_string();

        let h2_re = regex::Regex::new(r"(?m)^## (.+)$").unwrap();
        html = h2_re.replace_all(&html, "<h2>$1</h2>").to_string();

        let h3_re = regex::Regex::new(r"(?m)^### (.+)$").unwrap();
        html = h3_re.replace_all(&html, "<h3>$1</h3>").to_string();

        // Bold
        let bold_re = regex::Regex::new(r"\*\*(.+?)\*\*").unwrap();
        html = bold_re.replace_all(&html, "<strong>$1</strong>").to_string();

        // Italic
        let italic_re = regex::Regex::new(r"\*(.+?)\*").unwrap();
        html = italic_re.replace_all(&html, "<em>$1</em>").to_string();

        // Code blocks
        let code_block_re = regex::Regex::new(r"```(\w*)\n([\s\S]*?)```").unwrap();
        html = code_block_re
            .replace_all(&html, "<pre><code class=\"language-$1\">$2</code></pre>")
            .to_string();

        // Inline code
        let inline_code_re = regex::Regex::new(r"`([^`]+)`").unwrap();
        html = inline_code_re.replace_all(&html, "<code>$1</code>").to_string();

        // Links
        let link_re = regex::Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap();
        html = link_re.replace_all(&html, "<a href=\"$2\">$1</a>").to_string();

        // Paragraphs
        let lines: Vec<&str> = html.lines().collect();
        let mut result = Vec::new();
        for line in lines {
            let trimmed = line.trim();
            if !trimmed.is_empty()
                && !trimmed.starts_with('<')
                && !trimmed.starts_with('-')
                && !trimmed.starts_with('*')
            {
                result.push(format!("<p>{}</p>", trimmed));
            } else {
                result.push(line.to_string());
            }
        }

        Ok(result.join("\n"))
    }

    fn markdown_to_plaintext(&self, content: &str) -> Result<String> {
        let mut result = content.to_string();

        // Remove code blocks
        let code_block_re = regex::Regex::new(r"```[\s\S]*?```").unwrap();
        result = code_block_re.replace_all(&result, "").to_string();

        // Remove inline code
        let inline_code_re = regex::Regex::new(r"`([^`]+)`").unwrap();
        result = inline_code_re.replace_all(&result, "$1").to_string();

        // Remove headers
        let header_re = regex::Regex::new(r"(?m)^#{1,6}\s*").unwrap();
        result = header_re.replace_all(&result, "").to_string();

        // Remove bold
        let bold_re = regex::Regex::new(r"\*\*([^*]+)\*\*").unwrap();
        result = bold_re.replace_all(&result, "$1").to_string();

        // Remove italic
        let italic_re = regex::Regex::new(r"\*([^*]+)\*").unwrap();
        result = italic_re.replace_all(&result, "$1").to_string();

        // Remove links, keep text
        let link_re = regex::Regex::new(r"\[([^\]]+)\]\([^)]+\)").unwrap();
        result = link_re.replace_all(&result, "$1").to_string();

        // Remove list markers
        let list_re = regex::Regex::new(r"(?m)^[\s]*[-*+]\s").unwrap();
        result = list_re.replace_all(&result, "").to_string();

        Ok(result.trim().to_string())
    }

    fn markdown_to_latex(&self, content: &str) -> Result<String> {
        let mut latex = content.to_string();

        // Headers
        let h1_re = regex::Regex::new(r"(?m)^# (.+)$").unwrap();
        latex = h1_re.replace_all(&latex, "\\section{$1}").to_string();

        let h2_re = regex::Regex::new(r"(?m)^## (.+)$").unwrap();
        latex = h2_re.replace_all(&latex, "\\subsection{$1}").to_string();

        let h3_re = regex::Regex::new(r"(?m)^### (.+)$").unwrap();
        latex = h3_re.replace_all(&latex, "\\subsubsection{$1}").to_string();

        // Bold
        let bold_re = regex::Regex::new(r"\*\*(.+?)\*\*").unwrap();
        latex = bold_re.replace_all(&latex, "\\textbf{$1}").to_string();

        // Italic
        let italic_re = regex::Regex::new(r"\*(.+?)\*").unwrap();
        latex = italic_re.replace_all(&latex, "\\textit{$1}").to_string();

        // Code blocks
        let code_block_re = regex::Regex::new(r"```(\w*)\n([\s\S]*?)```").unwrap();
        latex = code_block_re
            .replace_all(&latex, "\\begin{lstlisting}[language=$1]\n$2\\end{lstlisting}")
            .to_string();

        // Inline code
        let inline_code_re = regex::Regex::new(r"`([^`]+)`").unwrap();
        latex = inline_code_re.replace_all(&latex, "\\texttt{$1}").to_string();

        Ok(latex)
    }

    fn markdown_to_json(&self, content: &str) -> Result<String> {
        let mut sections = Vec::new();
        let mut current_section: Option<(String, Vec<String>)> = None;

        for line in content.lines() {
            if line.starts_with('#') {
                // Save previous section
                if let Some((title, content)) = current_section.take() {
                    sections.push(serde_json::json!({
                        "title": title,
                        "content": content.join("\n")
                    }));
                }
                // Start new section
                let title = line.trim_start_matches('#').trim().to_string();
                current_section = Some((title, Vec::new()));
            } else if let Some((_, ref mut content)) = current_section {
                if !line.trim().is_empty() {
                    content.push(line.to_string());
                }
            }
        }

        // Save last section
        if let Some((title, content)) = current_section {
            sections.push(serde_json::json!({
                "title": title,
                "content": content.join("\n")
            }));
        }

        if sections.is_empty() {
            // No sections found, wrap as plain content
            let result = serde_json::json!({
                "content": content.trim()
            });
            if self.config.pretty_print {
                serde_json::to_string_pretty(&result).map_err(|e| Error::Synthesis(e.to_string()))
            } else {
                serde_json::to_string(&result).map_err(|e| Error::Synthesis(e.to_string()))
            }
        } else {
            let result = serde_json::json!({
                "sections": sections
            });
            if self.config.pretty_print {
                serde_json::to_string_pretty(&result).map_err(|e| Error::Synthesis(e.to_string()))
            } else {
                serde_json::to_string(&result).map_err(|e| Error::Synthesis(e.to_string()))
            }
        }
    }

    // =========================================================================
    // HTML CONVERSIONS
    // =========================================================================

    fn html_to_markdown(&self, content: &str) -> Result<String> {
        let mut md = content.to_string();

        // Headers
        let h1_re = regex::Regex::new(r"<h1>([^<]+)</h1>").unwrap();
        md = h1_re.replace_all(&md, "# $1").to_string();

        let h2_re = regex::Regex::new(r"<h2>([^<]+)</h2>").unwrap();
        md = h2_re.replace_all(&md, "## $1").to_string();

        let h3_re = regex::Regex::new(r"<h3>([^<]+)</h3>").unwrap();
        md = h3_re.replace_all(&md, "### $1").to_string();

        // Bold
        let strong_re = regex::Regex::new(r"<strong>([^<]+)</strong>").unwrap();
        md = strong_re.replace_all(&md, "**$1**").to_string();

        // Italic
        let em_re = regex::Regex::new(r"<em>([^<]+)</em>").unwrap();
        md = em_re.replace_all(&md, "*$1*").to_string();

        // Code
        let code_re = regex::Regex::new(r"<code>([^<]+)</code>").unwrap();
        md = code_re.replace_all(&md, "`$1`").to_string();

        // Links
        let link_re = regex::Regex::new(r#"<a href="([^"]+)">([^<]+)</a>"#).unwrap();
        md = link_re.replace_all(&md, "[$2]($1)").to_string();

        // Paragraphs
        let p_re = regex::Regex::new(r"<p>([^<]+)</p>").unwrap();
        md = p_re.replace_all(&md, "$1\n").to_string();

        // Remove remaining tags
        let tag_re = regex::Regex::new(r"<[^>]+>").unwrap();
        md = tag_re.replace_all(&md, "").to_string();

        Ok(md.trim().to_string())
    }

    fn html_to_plaintext(&self, content: &str) -> Result<String> {
        // Remove all HTML tags
        let tag_re = regex::Regex::new(r"<[^>]+>").unwrap();
        let result = tag_re.replace_all(content, "").to_string();

        // Decode common HTML entities
        let result = result
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&nbsp;", " ");

        Ok(result.trim().to_string())
    }

    fn html_to_latex(&self, content: &str) -> Result<String> {
        // First convert to markdown, then to latex
        let md = self.html_to_markdown(content)?;
        self.markdown_to_latex(&md)
    }

    fn html_to_json(&self, content: &str) -> Result<String> {
        // Wrap HTML content in JSON structure
        let result = serde_json::json!({
            "html": content.trim(),
            "plaintext": self.html_to_plaintext(content)?
        });
        if self.config.pretty_print {
            serde_json::to_string_pretty(&result).map_err(|e| Error::Synthesis(e.to_string()))
        } else {
            serde_json::to_string(&result).map_err(|e| Error::Synthesis(e.to_string()))
        }
    }

    // =========================================================================
    // XML CONVERSIONS
    // =========================================================================

    fn xml_to_json(&self, content: &str) -> Result<String> {
        // Simple XML to JSON conversion
        // For complex XML, a proper XML parser would be needed
        let result = serde_json::json!({
            "xml_content": content.trim()
        });
        if self.config.pretty_print {
            serde_json::to_string_pretty(&result).map_err(|e| Error::Synthesis(e.to_string()))
        } else {
            serde_json::to_string(&result).map_err(|e| Error::Synthesis(e.to_string()))
        }
    }

    fn xml_to_plaintext(&self, content: &str) -> Result<String> {
        // Remove XML tags
        let tag_re = regex::Regex::new(r"<[^>]+>").unwrap();
        let result = tag_re.replace_all(content, "").to_string();

        // Decode common XML entities
        let result = result
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'");

        Ok(result.trim().to_string())
    }

    fn xml_to_html(&self, content: &str) -> Result<String> {
        // XML is already HTML-like, just wrap in pre for display
        Ok(format!("<pre><code>{}</code></pre>", content))
    }

    // =========================================================================
    // PLAINTEXT CONVERSIONS
    // =========================================================================

    fn plaintext_to_json(&self, content: &str) -> Result<String> {
        let lines: Vec<&str> = content.lines().collect();
        let result = if lines.len() == 1 {
            serde_json::json!({ "content": content.trim() })
        } else {
            serde_json::json!({ "lines": lines })
        };
        if self.config.pretty_print {
            serde_json::to_string_pretty(&result).map_err(|e| Error::Synthesis(e.to_string()))
        } else {
            serde_json::to_string(&result).map_err(|e| Error::Synthesis(e.to_string()))
        }
    }

    fn plaintext_to_yaml(&self, content: &str) -> Result<String> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() == 1 {
            Ok(format!("content: {}", content.trim()))
        } else {
            let items: Vec<String> = lines.iter().map(|l| format!("- {}", l)).collect();
            Ok(format!("lines:\n{}", items.join("\n")))
        }
    }

    fn plaintext_to_markdown(&self, content: &str) -> Result<String> {
        Ok(content.to_string())
    }

    fn plaintext_to_html(&self, content: &str) -> Result<String> {
        let escaped = content
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        let paragraphs: Vec<String> = escaped
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .map(|p| format!("<p>{}</p>", p.trim()))
            .collect();
        Ok(paragraphs.join("\n"))
    }

    fn plaintext_to_xml(&self, content: &str) -> Result<String> {
        let escaped = content
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;");
        Ok(format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<content>{}</content>", escaped))
    }

    fn plaintext_to_latex(&self, content: &str) -> Result<String> {
        // Escape LaTeX special characters
        let escaped = content
            .replace('\\', "\\textbackslash{}")
            .replace('{', "\\{")
            .replace('}', "\\}")
            .replace('%', "\\%")
            .replace('$', "\\$")
            .replace('&', "\\&")
            .replace('#', "\\#")
            .replace('_', "\\_")
            .replace('^', "\\^{}")
            .replace('~', "\\~{}");
        Ok(escaped)
    }

    // =========================================================================
    // LATEX CONVERSIONS
    // =========================================================================

    fn latex_to_plaintext(&self, content: &str) -> Result<String> {
        let mut result = content.to_string();

        // Remove LaTeX commands
        let command_re = regex::Regex::new(r"\\[a-zA-Z]+\{([^}]*)\}").unwrap();
        result = command_re.replace_all(&result, "$1").to_string();

        // Remove remaining backslash commands
        let backslash_re = regex::Regex::new(r"\\[a-zA-Z]+").unwrap();
        result = backslash_re.replace_all(&result, "").to_string();

        // Remove braces
        result = result.replace('{', "").replace('}', "");

        Ok(result.trim().to_string())
    }

    fn latex_to_markdown(&self, content: &str) -> Result<String> {
        let mut md = content.to_string();

        // Sections
        let section_re = regex::Regex::new(r"\\section\{([^}]+)\}").unwrap();
        md = section_re.replace_all(&md, "# $1").to_string();

        let subsection_re = regex::Regex::new(r"\\subsection\{([^}]+)\}").unwrap();
        md = subsection_re.replace_all(&md, "## $1").to_string();

        let subsubsection_re = regex::Regex::new(r"\\subsubsection\{([^}]+)\}").unwrap();
        md = subsubsection_re.replace_all(&md, "### $1").to_string();

        // Text formatting
        let textbf_re = regex::Regex::new(r"\\textbf\{([^}]+)\}").unwrap();
        md = textbf_re.replace_all(&md, "**$1**").to_string();

        let textit_re = regex::Regex::new(r"\\textit\{([^}]+)\}").unwrap();
        md = textit_re.replace_all(&md, "*$1*").to_string();

        let texttt_re = regex::Regex::new(r"\\texttt\{([^}]+)\}").unwrap();
        md = texttt_re.replace_all(&md, "`$1`").to_string();

        Ok(md.trim().to_string())
    }

    fn latex_to_html(&self, content: &str) -> Result<String> {
        // First convert to markdown, then to HTML
        let md = self.latex_to_markdown(content)?;
        self.markdown_to_html(&md)
    }

    fn latex_to_json(&self, content: &str) -> Result<String> {
        let plaintext = self.latex_to_plaintext(content)?;
        let result = serde_json::json!({
            "latex": content.trim(),
            "plaintext": plaintext
        });
        if self.config.pretty_print {
            serde_json::to_string_pretty(&result).map_err(|e| Error::Synthesis(e.to_string()))
        } else {
            serde_json::to_string(&result).map_err(|e| Error::Synthesis(e.to_string()))
        }
    }

    // =========================================================================
    // CUSTOM FORMAT
    // =========================================================================

    fn apply_custom_template(&self, content: &str, template: &str) -> Result<String> {
        // Simple template substitution
        let result = template.replace("{content}", content);
        Ok(result)
    }

    // =========================================================================
    // VALUE CONVERTERS
    // =========================================================================

    fn value_to_xml(&self, value: &serde_json::Value, name: &str, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        match value {
            serde_json::Value::Null => format!("{}<{}/>\n", indent, name),
            serde_json::Value::Bool(b) => format!("{}<{}>{}</{}>\n", indent, name, b, name),
            serde_json::Value::Number(n) => format!("{}<{}>{}</{}>\n", indent, name, n, name),
            serde_json::Value::String(s) => {
                let escaped = s
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;");
                format!("{}<{}>{}</{}>\n", indent, name, escaped, name)
            }
            serde_json::Value::Array(arr) => {
                let mut result = format!("{}<{}>\n", indent, name);
                for item in arr {
                    result.push_str(&self.value_to_xml(item, "item", depth + 1));
                }
                result.push_str(&format!("{}</{}>\n", indent, name));
                result
            }
            serde_json::Value::Object(obj) => {
                let mut result = format!("{}<{}>\n", indent, name);
                for (key, val) in obj {
                    result.push_str(&self.value_to_xml(val, key, depth + 1));
                }
                result.push_str(&format!("{}</{}>\n", indent, name));
                result
            }
        }
    }

    fn value_to_markdown(&self, value: &serde_json::Value, depth: usize) -> String {
        let header_prefix = "#".repeat(depth.min(6));
        match value {
            serde_json::Value::Null => "null\n".to_string(),
            serde_json::Value::Bool(b) => format!("{}\n", b),
            serde_json::Value::Number(n) => format!("{}\n", n),
            serde_json::Value::String(s) => format!("{}\n", s),
            serde_json::Value::Array(arr) => {
                let mut result = String::new();
                for item in arr {
                    result.push_str("- ");
                    result.push_str(&self.value_to_markdown(item, depth));
                }
                result
            }
            serde_json::Value::Object(obj) => {
                let mut result = String::new();
                for (key, val) in obj {
                    result.push_str(&format!("{} {}\n\n", header_prefix, key));
                    result.push_str(&self.value_to_markdown(val, depth + 1));
                }
                result
            }
        }
    }

    fn value_to_plaintext(&self, value: &serde_json::Value, prefix: &str) -> String {
        match value {
            serde_json::Value::Null => "null\n".to_string(),
            serde_json::Value::Bool(b) => format!("{}\n", b),
            serde_json::Value::Number(n) => format!("{}\n", n),
            serde_json::Value::String(s) => format!("{}\n", s),
            serde_json::Value::Array(arr) => {
                let mut result = String::new();
                for (i, item) in arr.iter().enumerate() {
                    result.push_str(&format!("{}[{}]: ", prefix, i));
                    result.push_str(&self.value_to_plaintext(item, prefix));
                }
                result
            }
            serde_json::Value::Object(obj) => {
                let mut result = String::new();
                for (key, val) in obj {
                    result.push_str(&format!("{}{}: ", prefix, key));
                    result.push_str(&self.value_to_plaintext(val, &format!("{}  ", prefix)));
                }
                result
            }
        }
    }

    fn value_to_html(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Null => "<span class=\"null\">null</span>".to_string(),
            serde_json::Value::Bool(b) => format!("<span class=\"bool\">{}</span>", b),
            serde_json::Value::Number(n) => format!("<span class=\"number\">{}</span>", n),
            serde_json::Value::String(s) => {
                let escaped = s
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;");
                format!("<span class=\"string\">{}</span>", escaped)
            }
            serde_json::Value::Array(arr) => {
                let mut result = "<ul>\n".to_string();
                for item in arr {
                    result.push_str("<li>");
                    result.push_str(&self.value_to_html(item));
                    result.push_str("</li>\n");
                }
                result.push_str("</ul>");
                result
            }
            serde_json::Value::Object(obj) => {
                let mut result = "<dl>\n".to_string();
                for (key, val) in obj {
                    result.push_str(&format!("<dt>{}</dt>\n", key));
                    result.push_str("<dd>");
                    result.push_str(&self.value_to_html(val));
                    result.push_str("</dd>\n");
                }
                result.push_str("</dl>");
                result
            }
        }
    }

    fn value_to_latex(&self, value: &serde_json::Value, depth: usize) -> String {
        match value {
            serde_json::Value::Null => "\\textit{null}\n".to_string(),
            serde_json::Value::Bool(b) => format!("\\texttt{{{}}}\n", b),
            serde_json::Value::Number(n) => format!("{}\n", n),
            serde_json::Value::String(s) => {
                let escaped = s
                    .replace('\\', "\\textbackslash{}")
                    .replace('{', "\\{")
                    .replace('}', "\\}")
                    .replace('%', "\\%")
                    .replace('$', "\\$")
                    .replace('&', "\\&")
                    .replace('#', "\\#")
                    .replace('_', "\\_");
                format!("{}\n", escaped)
            }
            serde_json::Value::Array(arr) => {
                let mut result = "\\begin{itemize}\n".to_string();
                for item in arr {
                    result.push_str("\\item ");
                    result.push_str(&self.value_to_latex(item, depth));
                }
                result.push_str("\\end{itemize}\n");
                result
            }
            serde_json::Value::Object(obj) => {
                let mut result = String::new();
                let section_cmd = match depth {
                    1 => "section",
                    2 => "subsection",
                    _ => "subsubsection",
                };
                for (key, val) in obj {
                    result.push_str(&format!("\\{}{{{}}}\n", section_cmd, key));
                    result.push_str(&self.value_to_latex(val, depth + 1));
                }
                result
            }
        }
    }

    // =========================================================================
    // VALIDATION
    // =========================================================================

    /// Validate content against a target format.
    pub fn validate(&self, content: &str, format: &TargetFormat) -> ValidationResult {
        let mut result = ValidationResult::success();

        match format {
            TargetFormat::Json => {
                if let Err(e) = serde_json::from_str::<serde_json::Value>(content) {
                    result.add_error(format!("Invalid JSON: {}", e));
                }
            }
            TargetFormat::Yaml => {
                if let Err(e) = serde_yaml::from_str::<serde_yaml::Value>(content) {
                    result.add_error(format!("Invalid YAML: {}", e));
                }
            }
            TargetFormat::Xml => {
                if !self.looks_like_xml(content) {
                    result.add_error("Content does not appear to be valid XML");
                }
            }
            TargetFormat::Html => {
                if !content.contains('<') || !content.contains('>') {
                    result.add_warning("Content may not be valid HTML");
                }
            }
            TargetFormat::Markdown => {
                // Markdown is very permissive, just check for some structure
                if !self.looks_like_markdown(content) && content.len() > 100 {
                    result.add_warning("Content may benefit from Markdown formatting");
                }
            }
            TargetFormat::Latex => {
                if !content.contains('\\') {
                    result.add_warning("Content does not contain LaTeX commands");
                }
            }
            TargetFormat::PlainText => {
                // Plain text is always valid
            }
            TargetFormat::Custom(_) => {
                // Custom formats are assumed valid
            }
        }

        result
    }

    /// Generate a schema description for the target format.
    fn generate_schema(&self, format: &TargetFormat) -> String {
        match format {
            TargetFormat::Json => "JSON Schema: https://json-schema.org/".to_string(),
            TargetFormat::Yaml => "YAML 1.2 Specification".to_string(),
            TargetFormat::Xml => "XML 1.0 Specification".to_string(),
            TargetFormat::Markdown => "CommonMark Specification".to_string(),
            TargetFormat::Html => "HTML5 Specification".to_string(),
            TargetFormat::Latex => "LaTeX2e Documentation".to_string(),
            TargetFormat::PlainText => "UTF-8 Plain Text".to_string(),
            TargetFormat::Custom(name) => format!("Custom Format: {}", name),
        }
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom AdapterAgent configuration.
#[derive(Debug, Default)]
pub struct AdapterAgentBuilder {
    config: Option<AdapterConfig>,
    rules: Vec<TransformationRule>,
}

impl AdapterAgentBuilder {
    /// Set the configuration.
    pub fn config(mut self, config: AdapterConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Enable or disable structure preservation.
    pub fn preserve_structure(mut self, preserve: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.preserve_structure = preserve;
        self.config = Some(config);
        self
    }

    /// Enable or disable output validation.
    pub fn validate_output(mut self, validate: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.validate_output = validate;
        self.config = Some(config);
        self
    }

    /// Enable or disable pretty printing.
    pub fn pretty_print(mut self, pretty: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.pretty_print = pretty;
        self.config = Some(config);
        self
    }

    /// Set the maximum nesting depth.
    pub fn max_depth(mut self, depth: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_depth = depth;
        self.config = Some(config);
        self
    }

    /// Enable or disable bidirectional transformations.
    pub fn enable_bidirectional(mut self, enable: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.enable_bidirectional = enable;
        self.config = Some(config);
        self
    }

    /// Add a transformation rule.
    pub fn add_rule(mut self, rule: TransformationRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Build the AdapterAgent.
    pub fn build(self) -> AdapterAgent {
        let mut config = self.config.unwrap_or_default();
        config.rules = self.rules;
        config.rules.sort_by(|a, b| b.priority.cmp(&a.priority));

        AdapterAgent {
            config,
            format_rules: AdapterAgent::default_format_rules(),
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
    // TargetFormat Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_target_format_display() {
        assert_eq!(TargetFormat::Json.to_string(), "json");
        assert_eq!(TargetFormat::Yaml.to_string(), "yaml");
        assert_eq!(TargetFormat::Xml.to_string(), "xml");
        assert_eq!(TargetFormat::Markdown.to_string(), "markdown");
        assert_eq!(TargetFormat::PlainText.to_string(), "plaintext");
        assert_eq!(TargetFormat::Html.to_string(), "html");
        assert_eq!(TargetFormat::Latex.to_string(), "latex");
        assert_eq!(
            TargetFormat::Custom("template".to_string()).to_string(),
            "custom:template"
        );
    }

    #[test]
    fn test_target_format_from_str() {
        assert_eq!("json".parse::<TargetFormat>().unwrap(), TargetFormat::Json);
        assert_eq!("yaml".parse::<TargetFormat>().unwrap(), TargetFormat::Yaml);
        assert_eq!("yml".parse::<TargetFormat>().unwrap(), TargetFormat::Yaml);
        assert_eq!("xml".parse::<TargetFormat>().unwrap(), TargetFormat::Xml);
        assert_eq!("markdown".parse::<TargetFormat>().unwrap(), TargetFormat::Markdown);
        assert_eq!("md".parse::<TargetFormat>().unwrap(), TargetFormat::Markdown);
        assert_eq!("plaintext".parse::<TargetFormat>().unwrap(), TargetFormat::PlainText);
        assert_eq!("plain".parse::<TargetFormat>().unwrap(), TargetFormat::PlainText);
        assert_eq!("text".parse::<TargetFormat>().unwrap(), TargetFormat::PlainText);
        assert_eq!("html".parse::<TargetFormat>().unwrap(), TargetFormat::Html);
        assert_eq!("htm".parse::<TargetFormat>().unwrap(), TargetFormat::Html);
        assert_eq!("latex".parse::<TargetFormat>().unwrap(), TargetFormat::Latex);
        assert_eq!("tex".parse::<TargetFormat>().unwrap(), TargetFormat::Latex);
    }

    #[test]
    fn test_target_format_from_str_custom() {
        let result = "custom:my_template".parse::<TargetFormat>().unwrap();
        assert_eq!(result, TargetFormat::Custom("my_template".to_string()));
    }

    #[test]
    fn test_target_format_from_str_invalid() {
        assert!("invalid".parse::<TargetFormat>().is_err());
    }

    #[test]
    fn test_target_format_extension() {
        assert_eq!(TargetFormat::Json.extension(), "json");
        assert_eq!(TargetFormat::Yaml.extension(), "yaml");
        assert_eq!(TargetFormat::Xml.extension(), "xml");
        assert_eq!(TargetFormat::Markdown.extension(), "md");
        assert_eq!(TargetFormat::PlainText.extension(), "txt");
        assert_eq!(TargetFormat::Html.extension(), "html");
        assert_eq!(TargetFormat::Latex.extension(), "tex");
    }

    #[test]
    fn test_target_format_mime_type() {
        assert_eq!(TargetFormat::Json.mime_type(), "application/json");
        assert_eq!(TargetFormat::Html.mime_type(), "text/html");
        assert_eq!(TargetFormat::PlainText.mime_type(), "text/plain");
    }

    #[test]
    fn test_target_format_all_standard() {
        let all = TargetFormat::all_standard();
        assert_eq!(all.len(), 7);
        assert!(all.contains(&TargetFormat::Json));
        assert!(all.contains(&TargetFormat::Yaml));
    }

    // -------------------------------------------------------------------------
    // ValidationResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validation_result_success() {
        let result = ValidationResult::success();
        assert!(result.is_valid);
        assert!(!result.has_errors());
        assert!(!result.has_warnings());
        assert!((result.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validation_result_failure() {
        let result = ValidationResult::failure(vec!["Error 1".to_string(), "Error 2".to_string()]);
        assert!(!result.is_valid);
        assert!(result.has_errors());
        assert_eq!(result.errors.len(), 2);
        assert!((result.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validation_result_with_warnings() {
        let result = ValidationResult::success()
            .with_warnings(vec!["Warning 1".to_string()]);
        assert!(result.is_valid);
        assert!(result.has_warnings());
        assert!(result.confidence <= 0.8);
    }

    #[test]
    fn test_validation_result_add_error() {
        let mut result = ValidationResult::success();
        result.add_error("New error");
        assert!(!result.is_valid);
        assert!(result.has_errors());
    }

    #[test]
    fn test_validation_result_add_warning() {
        let mut result = ValidationResult::success();
        result.add_warning("New warning");
        assert!(result.is_valid);
        assert!(result.has_warnings());
    }

    // -------------------------------------------------------------------------
    // TransformationRule Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transformation_rule_new() {
        let rule = TransformationRule::new("pattern", "replacement");
        assert_eq!(rule.source_pattern, "pattern");
        assert_eq!(rule.target_pattern, "replacement");
        assert_eq!(rule.priority, 0);
        assert!(!rule.bidirectional);
    }

    #[test]
    fn test_transformation_rule_builder() {
        let rule = TransformationRule::new("source", "target")
            .with_priority(10)
            .with_description("Test rule")
            .bidirectional();

        assert_eq!(rule.priority, 10);
        assert_eq!(rule.description, Some("Test rule".to_string()));
        assert!(rule.bidirectional);
    }

    #[test]
    fn test_transformation_rule_apply() {
        let rule = TransformationRule::new("hello", "world");
        let result = rule.apply("say hello");
        assert_eq!(result, "say world");
    }

    #[test]
    fn test_transformation_rule_apply_regex() {
        let rule = TransformationRule::new(r"\d+", "NUMBER");
        let result = rule.apply("I have 42 apples");
        assert_eq!(result, "I have NUMBER apples");
    }

    #[test]
    fn test_transformation_rule_apply_reverse() {
        let rule = TransformationRule::new("hello", "world").bidirectional();
        let result = rule.apply_reverse("say world").unwrap();
        assert_eq!(result, "say hello");
    }

    #[test]
    fn test_transformation_rule_apply_reverse_not_bidirectional() {
        let rule = TransformationRule::new("hello", "world");
        let result = rule.apply_reverse("say world");
        assert!(result.is_none());
    }

    // -------------------------------------------------------------------------
    // AdapterConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_adapter_config_default() {
        let config = AdapterConfig::default();
        assert!(config.preserve_structure);
        assert!(config.validate_output);
        assert!(!config.include_schema);
        assert!(config.pretty_print);
        assert_eq!(config.max_depth, 10);
    }

    #[test]
    fn test_adapter_config_machine() {
        let config = AdapterConfig::machine();
        assert!(!config.pretty_print);
        assert!(config.include_schema);
        assert_eq!(config.max_depth, 20);
    }

    #[test]
    fn test_adapter_config_human() {
        let config = AdapterConfig::human();
        assert!(config.pretty_print);
        assert!(!config.validate_output);
        assert_eq!(config.max_depth, 5);
    }

    #[test]
    fn test_adapter_config_add_rule() {
        let rule = TransformationRule::new("a", "b").with_priority(5);
        let config = AdapterConfig::default().add_rule(rule);
        assert_eq!(config.rules.len(), 1);
    }

    // -------------------------------------------------------------------------
    // AdaptedPrompt Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_adapted_prompt_new() {
        let adapted = AdaptedPrompt::new("test content", TargetFormat::Json);
        assert_eq!(adapted.content, "test content");
        assert_eq!(adapted.original, "test content");
        assert_eq!(adapted.format, TargetFormat::Json);
        assert!(adapted.is_valid());
    }

    #[test]
    fn test_adapted_prompt_was_transformed() {
        let mut adapted = AdaptedPrompt::new("original", TargetFormat::Json);
        assert!(!adapted.was_transformed());

        adapted.content = "modified".to_string();
        assert!(adapted.was_transformed());
    }

    #[test]
    fn test_adapted_prompt_size_delta() {
        let mut adapted = AdaptedPrompt::new("abc", TargetFormat::Json);
        adapted.content = "abcdef".to_string();
        assert_eq!(adapted.size_delta(), 3);
    }

    #[test]
    fn test_adapted_prompt_with_source_format() {
        let adapted = AdaptedPrompt::new("test", TargetFormat::Json)
            .with_source_format(TargetFormat::Yaml);
        assert_eq!(adapted.source_format, Some(TargetFormat::Yaml));
    }

    #[test]
    fn test_adapted_prompt_with_schema() {
        let adapted = AdaptedPrompt::new("test", TargetFormat::Json)
            .with_schema("JSON Schema");
        assert_eq!(adapted.schema, Some("JSON Schema".to_string()));
    }

    // -------------------------------------------------------------------------
    // AdapterAgent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_adapter_agent_new() {
        let agent = AdapterAgent::new();
        assert!(agent.config.preserve_structure);
    }

    #[test]
    fn test_adapter_agent_builder() {
        let agent = AdapterAgent::builder()
            .preserve_structure(false)
            .validate_output(true)
            .pretty_print(false)
            .max_depth(5)
            .build();

        assert!(!agent.config.preserve_structure);
        assert!(agent.config.validate_output);
        assert!(!agent.config.pretty_print);
        assert_eq!(agent.config.max_depth, 5);
    }

    #[test]
    fn test_adapter_agent_detect_format_json() {
        let agent = AdapterAgent::new();
        let content = r#"{"key": "value"}"#;
        assert_eq!(agent.detect_format(content), TargetFormat::Json);
    }

    #[test]
    fn test_adapter_agent_detect_format_json_array() {
        let agent = AdapterAgent::new();
        let content = r#"[1, 2, 3]"#;
        assert_eq!(agent.detect_format(content), TargetFormat::Json);
    }

    #[test]
    fn test_adapter_agent_detect_format_yaml() {
        let agent = AdapterAgent::new();
        let content = "---\nname: test\nvalue: 42";
        assert_eq!(agent.detect_format(content), TargetFormat::Yaml);
    }

    #[test]
    fn test_adapter_agent_detect_format_yaml_keyvalue() {
        let agent = AdapterAgent::new();
        let content = "name: test\nvalue: 42\nlist:\n  - item1";
        assert_eq!(agent.detect_format(content), TargetFormat::Yaml);
    }

    #[test]
    fn test_adapter_agent_detect_format_markdown() {
        let agent = AdapterAgent::new();
        let content = "# Header\n\nSome **bold** text\n\n- List item";
        assert_eq!(agent.detect_format(content), TargetFormat::Markdown);
    }

    #[test]
    fn test_adapter_agent_detect_format_html() {
        let agent = AdapterAgent::new();
        let content = "<!DOCTYPE html><html><body>Test</body></html>";
        assert_eq!(agent.detect_format(content), TargetFormat::Html);
    }

    #[test]
    fn test_adapter_agent_detect_format_xml() {
        let agent = AdapterAgent::new();
        let content = "<?xml version=\"1.0\"?><root><item>test</item></root>";
        assert_eq!(agent.detect_format(content), TargetFormat::Xml);
    }

    #[test]
    fn test_adapter_agent_detect_format_latex() {
        let agent = AdapterAgent::new();
        let content = "\\documentclass{article}\n\\begin{document}Test\\end{document}";
        assert_eq!(agent.detect_format(content), TargetFormat::Latex);
    }

    #[test]
    fn test_adapter_agent_detect_format_plaintext() {
        let agent = AdapterAgent::new();
        let content = "Just some plain text without any formatting.";
        assert_eq!(agent.detect_format(content), TargetFormat::PlainText);
    }

    #[test]
    fn test_adapter_agent_detect_format_empty() {
        let agent = AdapterAgent::new();
        assert_eq!(agent.detect_format(""), TargetFormat::PlainText);
    }

    // -------------------------------------------------------------------------
    // JSON Conversion Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_adapt_json_to_yaml() {
        let agent = AdapterAgent::new();
        let json = r#"{"name": "test", "value": 42}"#;
        let result = agent.adapt(json, TargetFormat::Yaml).await.unwrap();
        assert!(result.content.contains("name:"));
        assert!(result.content.contains("value:"));
    }

    #[tokio::test]
    async fn test_adapt_json_to_xml() {
        let agent = AdapterAgent::new();
        let json = r#"{"name": "test"}"#;
        let result = agent.adapt(json, TargetFormat::Xml).await.unwrap();
        assert!(result.content.contains("<root>"));
        assert!(result.content.contains("<name>"));
    }

    #[tokio::test]
    async fn test_adapt_json_to_markdown() {
        let agent = AdapterAgent::new();
        let json = r#"{"title": "Test"}"#;
        let result = agent.adapt(json, TargetFormat::Markdown).await.unwrap();
        assert!(result.content.contains("#"));
    }

    #[tokio::test]
    async fn test_adapt_json_to_html() {
        let agent = AdapterAgent::new();
        let json = r#"{"key": "value"}"#;
        let result = agent.adapt(json, TargetFormat::Html).await.unwrap();
        assert!(result.content.contains("<dl>"));
    }

    // -------------------------------------------------------------------------
    // YAML Conversion Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_adapt_yaml_to_json() {
        let agent = AdapterAgent::new();
        let yaml = "name: test\nvalue: 42";
        let result = agent.adapt(yaml, TargetFormat::Json).await.unwrap();
        assert!(result.content.contains("\"name\""));
        assert!(result.content.contains("\"value\""));
    }

    // -------------------------------------------------------------------------
    // Markdown Conversion Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_adapt_markdown_to_html() {
        let agent = AdapterAgent::new();
        let md = "# Header\n\n**Bold** and *italic*";
        let result = agent.adapt(md, TargetFormat::Html).await.unwrap();
        assert!(result.content.contains("<h1>Header</h1>"));
        assert!(result.content.contains("<strong>Bold</strong>"));
        assert!(result.content.contains("<em>italic</em>"));
    }

    #[tokio::test]
    async fn test_adapt_markdown_to_plaintext() {
        let agent = AdapterAgent::new();
        let md = "# Header\n\n**Bold** text";
        let result = agent.adapt(md, TargetFormat::PlainText).await.unwrap();
        assert!(!result.content.contains('#'));
        assert!(!result.content.contains("**"));
    }

    #[tokio::test]
    async fn test_adapt_markdown_to_latex() {
        let agent = AdapterAgent::new();
        let md = "# Header\n\n**Bold** text";
        let result = agent.adapt(md, TargetFormat::Latex).await.unwrap();
        assert!(result.content.contains("\\section{Header}"));
        assert!(result.content.contains("\\textbf{Bold}"));
    }

    // -------------------------------------------------------------------------
    // HTML Conversion Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_adapt_html_to_markdown() {
        let agent = AdapterAgent::new();
        let html = "<h1>Header</h1><p><strong>Bold</strong></p>";
        let result = agent.adapt(html, TargetFormat::Markdown).await.unwrap();
        assert!(result.content.contains("# Header"));
        assert!(result.content.contains("**Bold**"));
    }

    #[tokio::test]
    async fn test_adapt_html_to_plaintext() {
        let agent = AdapterAgent::new();
        let html = "<h1>Header</h1><p>Text</p>";
        let result = agent.adapt(html, TargetFormat::PlainText).await.unwrap();
        assert!(!result.content.contains('<'));
        assert!(result.content.contains("Header"));
        assert!(result.content.contains("Text"));
    }

    // -------------------------------------------------------------------------
    // PlainText Conversion Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_adapt_plaintext_to_json() {
        let agent = AdapterAgent::new();
        let text = "Simple text";
        let result = agent.adapt(text, TargetFormat::Json).await.unwrap();
        assert!(result.content.contains("\"content\""));
    }

    #[tokio::test]
    async fn test_adapt_plaintext_to_html() {
        let agent = AdapterAgent::new();
        let text = "Paragraph 1\n\nParagraph 2";
        let result = agent.adapt(text, TargetFormat::Html).await.unwrap();
        assert!(result.content.contains("<p>"));
    }

    #[tokio::test]
    async fn test_adapt_plaintext_to_xml() {
        let agent = AdapterAgent::new();
        let text = "Simple text";
        let result = agent.adapt(text, TargetFormat::Xml).await.unwrap();
        assert!(result.content.contains("<?xml"));
        assert!(result.content.contains("<content>"));
    }

    // -------------------------------------------------------------------------
    // Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_json_valid() {
        let agent = AdapterAgent::new();
        let content = r#"{"key": "value"}"#;
        let result = agent.validate(content, &TargetFormat::Json);
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_json_invalid() {
        let agent = AdapterAgent::new();
        let content = "not valid json";
        let result = agent.validate(content, &TargetFormat::Json);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_yaml_valid() {
        let agent = AdapterAgent::new();
        let content = "key: value";
        let result = agent.validate(content, &TargetFormat::Yaml);
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_yaml_invalid() {
        let agent = AdapterAgent::new();
        let content = "key: [invalid: yaml";
        let result = agent.validate(content, &TargetFormat::Yaml);
        assert!(!result.is_valid);
    }

    // -------------------------------------------------------------------------
    // Error Handling Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_adapt_empty_content() {
        let agent = AdapterAgent::new();
        let result = agent.adapt("", TargetFormat::Json).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_adapt_whitespace_only() {
        let agent = AdapterAgent::new();
        let result = agent.adapt("   \n\t  ", TargetFormat::Json).await;
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Bidirectional Transformation Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_reverse_adapt() {
        let agent = AdapterAgent::new();
        let json = r#"{"name": "test"}"#;

        // First convert JSON to YAML
        let yaml_result = agent.adapt(json, TargetFormat::Yaml).await.unwrap();

        // Then convert back to JSON
        let json_result = agent
            .reverse_adapt(&yaml_result.content, TargetFormat::Json, TargetFormat::Yaml)
            .await
            .unwrap();

        assert!(json_result.content.contains("\"name\""));
    }

    #[tokio::test]
    async fn test_reverse_adapt_disabled() {
        let agent = AdapterAgent::builder()
            .enable_bidirectional(false)
            .build();

        let result = agent
            .reverse_adapt("content", TargetFormat::Json, TargetFormat::Yaml)
            .await;

        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Same Format Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_adapt_same_format() {
        let agent = AdapterAgent::new();
        let json = r#"{"key": "value"}"#;
        let result = agent.adapt(json, TargetFormat::Json).await.unwrap();
        assert_eq!(result.content, json);
    }

    // -------------------------------------------------------------------------
    // Custom Format Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_adapt_to_custom() {
        let agent = AdapterAgent::new();
        let content = "test content";
        let result = agent
            .adapt(content, TargetFormat::Custom("[{content}]".to_string()))
            .await
            .unwrap();
        assert_eq!(result.content, "[test content]");
    }

    // -------------------------------------------------------------------------
    // Builder with Rules Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_add_rule() {
        let rule = TransformationRule::new("test", "replaced");
        let agent = AdapterAgent::builder()
            .add_rule(rule)
            .build();
        assert_eq!(agent.config.rules.len(), 1);
    }

    #[test]
    fn test_builder_multiple_rules_sorted() {
        let rule1 = TransformationRule::new("a", "b").with_priority(1);
        let rule2 = TransformationRule::new("c", "d").with_priority(10);
        let agent = AdapterAgent::builder()
            .add_rule(rule1)
            .add_rule(rule2)
            .build();

        assert_eq!(agent.config.rules.len(), 2);
        assert_eq!(agent.config.rules[0].priority, 10); // Higher priority first
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_full_json_yaml_roundtrip() {
        let agent = AdapterAgent::new();
        let original_json = r#"{"name": "test", "value": 42, "items": [1, 2, 3]}"#;

        // JSON -> YAML
        let yaml = agent.adapt(original_json, TargetFormat::Yaml).await.unwrap();
        assert!(yaml.is_valid());
        assert!(yaml.was_transformed());

        // YAML -> JSON
        let back_to_json = agent.adapt(&yaml.content, TargetFormat::Json).await.unwrap();
        assert!(back_to_json.is_valid());

        // Verify roundtrip
        let original: serde_json::Value = serde_json::from_str(original_json).unwrap();
        let roundtripped: serde_json::Value = serde_json::from_str(&back_to_json.content).unwrap();
        assert_eq!(original, roundtripped);
    }

    #[tokio::test]
    async fn test_full_markdown_html_roundtrip() {
        let agent = AdapterAgent::new();
        let original_md = "# Header\n\n**Bold** text";

        // Markdown -> HTML
        let html = agent.adapt(original_md, TargetFormat::Html).await.unwrap();
        assert!(html.content.contains("<h1>"));
        assert!(html.content.contains("<strong>"));

        // HTML -> Markdown
        let back_to_md = agent.adapt(&html.content, TargetFormat::Markdown).await.unwrap();
        assert!(back_to_md.content.contains("# Header"));
        assert!(back_to_md.content.contains("**Bold**"));
    }
}
