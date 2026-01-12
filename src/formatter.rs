//! Formatter Agent module for Project Panpsychism.
//!
//! The Scribe - "Every spell must be inscribed in the proper form."
//!
//! This module implements the Formatter Agent, responsible for transforming
//! raw magical output into beautifully formatted manuscripts. Like a master
//! scribe in the sorcerer's tower, The Scribe understands many languages
//! and can translate between them seamlessly.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, The Scribe serves a crucial role:
//!
//! - **Raw Magic** (unformatted content) enters the scriptorium
//! - **The Scribe** (FormatterAgent) applies the appropriate runes
//! - **Inscribed Spell** (formatted output) emerges ready for use
//!
//! The Scribe can:
//! - Detect the nature of incoming magic (auto-detection)
//! - Transform between different runic systems (format conversion)
//! - Apply the sorcerer's preferred inscription style (templates)
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//! - **RATIO**: Logical structure in every format transformation
//! - **LAETITIA**: Joy through clarity and readability
//! - **NATURA**: Natural alignment between content and presentation
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::formatter::{FormatterAgent, OutputFormat};
//!
//! let scribe = FormatterAgent::new();
//!
//! // Transform raw magic into markdown
//! let formatted = scribe.format("My list: item1, item2, item3", OutputFormat::Bullet);
//! println!("{}", formatted.content);
//!
//! // Let The Scribe detect the format
//! let json_text = r#"{"key": "value"}"#;
//! let detected = scribe.auto_format(json_text);
//! println!("Detected as: {:?}", detected.format);
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// OUTPUT FORMAT ENUM
// =============================================================================

/// The runic systems known to The Scribe.
///
/// Each format represents a different way of inscribing magical output,
/// suitable for different purposes and consumers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Markdown - The universal language of documentation.
    ///
    /// Rich text with headers, emphasis, lists, and code blocks.
    /// Ideal for documentation, READMEs, and human-readable output.
    Markdown,

    /// JSON - Structured data in JavaScript Object Notation.
    ///
    /// Machine-readable, supports nested structures.
    /// Ideal for APIs, configuration, and data exchange.
    Json,

    /// YAML - Human-friendly data serialization.
    ///
    /// Similar to JSON but more readable for humans.
    /// Ideal for configuration files and readable data.
    Yaml,

    /// PlainText - Raw, unformatted text.
    ///
    /// No formatting, just the essence of the content.
    /// Ideal for simple output, logs, and terminal display.
    PlainText,

    /// Code - Source code in a specific programming language.
    ///
    /// Syntax-aware formatting with proper indentation.
    /// The `language` field specifies which language.
    Code {
        /// The programming language (e.g., "rust", "python", "javascript")
        language: String,
    },

    /// Table - Tabular data presentation.
    ///
    /// Data organized in rows and columns.
    /// Ideal for comparison, data sets, and structured output.
    Table,

    /// Bullet - Bulleted list format.
    ///
    /// Items as a hierarchical list with bullet points.
    /// Ideal for enumerations, steps, and collections.
    Bullet,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Markdown
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Markdown => write!(f, "markdown"),
            Self::Json => write!(f, "json"),
            Self::Yaml => write!(f, "yaml"),
            Self::PlainText => write!(f, "plain"),
            Self::Code { language } => write!(f, "code:{}", language),
            Self::Table => write!(f, "table"),
            Self::Bullet => write!(f, "bullet"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let lower = s.to_lowercase();

        // Handle code:language format
        if lower.starts_with("code:") {
            let language = lower.strip_prefix("code:").unwrap_or("text").to_string();
            return Ok(Self::Code { language });
        }

        match lower.as_str() {
            "markdown" | "md" => Ok(Self::Markdown),
            "json" => Ok(Self::Json),
            "yaml" | "yml" => Ok(Self::Yaml),
            "plain" | "plaintext" | "text" | "txt" => Ok(Self::PlainText),
            "table" | "tabular" => Ok(Self::Table),
            "bullet" | "bullets" | "list" => Ok(Self::Bullet),
            _ => Err(Error::Config(format!(
                "Unknown output format: '{}'. Valid formats: markdown, json, yaml, plain, \
                 code:<language>, table, bullet",
                s
            ))),
        }
    }
}

// =============================================================================
// FORMATTED OUTPUT
// =============================================================================

/// The result of The Scribe's inscription work.
///
/// Contains the formatted content along with metadata about
/// the transformation process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattedOutput {
    /// The formatted content string.
    pub content: String,

    /// The format that was applied.
    pub format: OutputFormat,

    /// Original content length before formatting.
    pub original_length: usize,

    /// Formatted content length.
    pub formatted_length: usize,

    /// Whether the format was auto-detected.
    pub auto_detected: bool,

    /// Confidence score for auto-detection (0.0 - 1.0).
    pub detection_confidence: f64,

    /// Any warnings generated during formatting.
    pub warnings: Vec<String>,
}

impl FormattedOutput {
    /// Create a new FormattedOutput.
    pub fn new(content: impl Into<String>, format: OutputFormat) -> Self {
        let content = content.into();
        let len = content.len();
        Self {
            content,
            format,
            original_length: len,
            formatted_length: len,
            auto_detected: false,
            detection_confidence: 1.0,
            warnings: Vec::new(),
        }
    }

    /// Check if the formatting process generated warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get the size change from formatting (positive = grew, negative = shrank).
    pub fn size_delta(&self) -> isize {
        self.formatted_length as isize - self.original_length as isize
    }

    /// Check if the content changed during formatting.
    pub fn was_transformed(&self) -> bool {
        self.original_length != self.formatted_length
    }
}

// =============================================================================
// FORMAT TEMPLATES
// =============================================================================

/// Template for formatting output in a specific format.
///
/// Templates define the structure and style of formatted output,
/// like the particular style of runes a scribe might use.
#[derive(Debug, Clone)]
pub struct FormatTemplate {
    /// The target format this template produces.
    pub format: OutputFormat,

    /// Header/prefix to add before content.
    pub header: Option<String>,

    /// Footer/suffix to add after content.
    pub footer: Option<String>,

    /// Wrapper for individual items (for list-like content).
    pub item_prefix: String,

    /// Suffix for individual items.
    pub item_suffix: String,

    /// Separator between items.
    pub separator: String,

    /// Indentation string (for nested content).
    pub indent: String,
}

impl Default for FormatTemplate {
    fn default() -> Self {
        Self {
            format: OutputFormat::PlainText,
            header: None,
            footer: None,
            item_prefix: String::new(),
            item_suffix: String::new(),
            separator: "\n".to_string(),
            indent: "  ".to_string(),
        }
    }
}

impl FormatTemplate {
    /// Create a template for Markdown format.
    pub fn markdown() -> Self {
        Self {
            format: OutputFormat::Markdown,
            header: None,
            footer: None,
            item_prefix: String::new(),
            item_suffix: String::new(),
            separator: "\n\n".to_string(),
            indent: "  ".to_string(),
        }
    }

    /// Create a template for JSON format.
    pub fn json() -> Self {
        Self {
            format: OutputFormat::Json,
            header: Some("{".to_string()),
            footer: Some("}".to_string()),
            item_prefix: "  ".to_string(),
            item_suffix: String::new(),
            separator: ",\n".to_string(),
            indent: "  ".to_string(),
        }
    }

    /// Create a template for YAML format.
    pub fn yaml() -> Self {
        Self {
            format: OutputFormat::Yaml,
            header: Some("---".to_string()),
            footer: None,
            item_prefix: String::new(),
            item_suffix: String::new(),
            separator: "\n".to_string(),
            indent: "  ".to_string(),
        }
    }

    /// Create a template for bullet list format.
    pub fn bullet() -> Self {
        Self {
            format: OutputFormat::Bullet,
            header: None,
            footer: None,
            item_prefix: "- ".to_string(),
            item_suffix: String::new(),
            separator: "\n".to_string(),
            indent: "  ".to_string(),
        }
    }

    /// Create a template for table format.
    pub fn table() -> Self {
        Self {
            format: OutputFormat::Table,
            header: None,
            footer: None,
            item_prefix: "| ".to_string(),
            item_suffix: " |".to_string(),
            separator: "\n".to_string(),
            indent: String::new(),
        }
    }

    /// Create a template for code format.
    pub fn code(language: &str) -> Self {
        Self {
            format: OutputFormat::Code {
                language: language.to_string(),
            },
            header: Some(format!("```{}", language)),
            footer: Some("```".to_string()),
            item_prefix: String::new(),
            item_suffix: String::new(),
            separator: "\n".to_string(),
            indent: "    ".to_string(),
        }
    }
}

// =============================================================================
// DETECTION PATTERNS
// =============================================================================

/// Regex patterns for format auto-detection.
///
/// Note: Debug implementation is manual since regex::Regex doesn't implement Debug
/// in a useful way for our purposes.
struct DetectionPatterns {
    json_object: regex::Regex,
    json_array: regex::Regex,
    yaml_header: regex::Regex,
    yaml_key_value: regex::Regex,
    markdown_header: regex::Regex,
    markdown_list: regex::Regex,
    markdown_code_block: regex::Regex,
    code_function: regex::Regex,
    code_class: regex::Regex,
    table_row: regex::Regex,
    bullet_item: regex::Regex,
}

impl std::fmt::Debug for DetectionPatterns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetectionPatterns")
            .field("json_object", &"<regex>")
            .field("json_array", &"<regex>")
            .field("yaml_header", &"<regex>")
            .field("yaml_key_value", &"<regex>")
            .field("markdown_header", &"<regex>")
            .field("markdown_list", &"<regex>")
            .field("markdown_code_block", &"<regex>")
            .field("code_function", &"<regex>")
            .field("code_class", &"<regex>")
            .field("table_row", &"<regex>")
            .field("bullet_item", &"<regex>")
            .finish()
    }
}

impl Default for DetectionPatterns {
    fn default() -> Self {
        Self {
            json_object: regex::Regex::new(r"^\s*\{[\s\S]*\}\s*$").expect("Invalid regex"),
            json_array: regex::Regex::new(r"^\s*\[[\s\S]*\]\s*$").expect("Invalid regex"),
            yaml_header: regex::Regex::new(r"^---\s*$").expect("Invalid regex"),
            yaml_key_value: regex::Regex::new(r"^\s*[\w-]+:\s*.+").expect("Invalid regex"),
            markdown_header: regex::Regex::new(r"^#{1,6}\s+.+").expect("Invalid regex"),
            markdown_list: regex::Regex::new(r"^[\s]*[-*+]\s+.+").expect("Invalid regex"),
            markdown_code_block: regex::Regex::new(r"^```[\w]*\s*$").expect("Invalid regex"),
            code_function: regex::Regex::new(r"(?:fn|function|def|func)\s+\w+\s*\(")
                .expect("Invalid regex"),
            code_class: regex::Regex::new(r"(?:class|struct|interface|trait)\s+\w+")
                .expect("Invalid regex"),
            table_row: regex::Regex::new(r"^\|[^|]+\|").expect("Invalid regex"),
            bullet_item: regex::Regex::new(r"^[\s]*[-*+]\s").expect("Invalid regex"),
        }
    }
}

// =============================================================================
// FORMATTER AGENT (THE SCRIBE)
// =============================================================================

/// The Formatter Agent - The Scribe of the Sorcerer's Tower.
///
/// Responsible for all output formatting, format detection, and conversion.
/// The Scribe ensures that every piece of magical output is properly
/// inscribed in the most appropriate runic system.
///
/// # Example
///
/// ```rust,ignore
/// let scribe = FormatterAgent::new();
///
/// // Format content as Markdown
/// let result = scribe.format("Hello World", OutputFormat::Markdown);
///
/// // Auto-detect and format
/// let result = scribe.auto_format(r#"{"key": "value"}"#);
/// assert_eq!(result.format, OutputFormat::Json);
///
/// // Convert between formats
/// let json = r#"{"name": "test", "value": 42}"#;
/// let yaml = scribe.convert(json, OutputFormat::Json, OutputFormat::Yaml)?;
/// ```
#[derive(Debug)]
pub struct FormatterAgent {
    /// Templates for each format type.
    templates: HashMap<String, FormatTemplate>,

    /// Compiled detection patterns.
    patterns: DetectionPatterns,

    /// Default format when auto-detection fails.
    default_format: OutputFormat,

    /// Minimum confidence threshold for auto-detection.
    detection_threshold: f64,
}

impl Default for FormatterAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatterAgent {
    /// Create a new Formatter Agent with default settings.
    ///
    /// The Scribe awakens, ready to inscribe your magical output.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let scribe = FormatterAgent::new();
    /// ```
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        templates.insert("markdown".to_string(), FormatTemplate::markdown());
        templates.insert("json".to_string(), FormatTemplate::json());
        templates.insert("yaml".to_string(), FormatTemplate::yaml());
        templates.insert("bullet".to_string(), FormatTemplate::bullet());
        templates.insert("table".to_string(), FormatTemplate::table());

        Self {
            templates,
            patterns: DetectionPatterns::default(),
            default_format: OutputFormat::PlainText,
            detection_threshold: 0.5,
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> FormatterAgentBuilder {
        FormatterAgentBuilder::default()
    }

    /// Get the default format.
    pub fn default_format(&self) -> &OutputFormat {
        &self.default_format
    }

    /// Get the detection threshold.
    pub fn detection_threshold(&self) -> f64 {
        self.detection_threshold
    }

    // =========================================================================
    // CORE FORMATTING METHODS
    // =========================================================================

    /// Format content in the specified output format.
    ///
    /// The Scribe inscribes the raw magic into the requested runic system.
    ///
    /// # Arguments
    ///
    /// * `content` - The raw content to format
    /// * `format` - The target output format
    ///
    /// # Returns
    ///
    /// A `FormattedOutput` containing the formatted content and metadata.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let scribe = FormatterAgent::new();
    /// let result = scribe.format("item1, item2, item3", OutputFormat::Bullet);
    /// println!("{}", result.content);
    /// // Output:
    /// // - item1
    /// // - item2
    /// // - item3
    /// ```
    pub fn format(&self, content: &str, format: OutputFormat) -> FormattedOutput {
        let original_length = content.len();

        let formatted = match &format {
            OutputFormat::Markdown => self.format_as_markdown(content),
            OutputFormat::Json => self.format_as_json(content),
            OutputFormat::Yaml => self.format_as_yaml(content),
            OutputFormat::PlainText => self.format_as_plain(content),
            OutputFormat::Code { language } => self.format_as_code(content, language),
            OutputFormat::Table => self.format_as_table(content),
            OutputFormat::Bullet => self.format_as_bullet(content),
        };

        FormattedOutput {
            formatted_length: formatted.len(),
            content: formatted,
            format,
            original_length,
            auto_detected: false,
            detection_confidence: 1.0,
            warnings: Vec::new(),
        }
    }

    /// Auto-detect the format and apply appropriate formatting.
    ///
    /// The Scribe examines the magical essence and determines
    /// the most suitable runic system for inscription.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to analyze and format
    ///
    /// # Returns
    ///
    /// A `FormattedOutput` with the detected format and formatted content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let scribe = FormatterAgent::new();
    ///
    /// let json = r#"{"key": "value"}"#;
    /// let result = scribe.auto_format(json);
    /// assert_eq!(result.format, OutputFormat::Json);
    /// assert!(result.auto_detected);
    /// ```
    pub fn auto_format(&self, content: &str) -> FormattedOutput {
        let (format, confidence) = self.detect_format(content);
        let original_length = content.len();

        // If confidence is below threshold, use default format
        let actual_format = if confidence >= self.detection_threshold {
            format
        } else {
            self.default_format.clone()
        };

        let formatted = match &actual_format {
            OutputFormat::Markdown => self.format_as_markdown(content),
            OutputFormat::Json => self.format_as_json(content),
            OutputFormat::Yaml => self.format_as_yaml(content),
            OutputFormat::PlainText => self.format_as_plain(content),
            OutputFormat::Code { language } => self.format_as_code(content, language),
            OutputFormat::Table => self.format_as_table(content),
            OutputFormat::Bullet => self.format_as_bullet(content),
        };

        FormattedOutput {
            formatted_length: formatted.len(),
            content: formatted,
            format: actual_format,
            original_length,
            auto_detected: true,
            detection_confidence: confidence,
            warnings: Vec::new(),
        }
    }

    /// Convert content from one format to another.
    ///
    /// The Scribe translates between different runic systems,
    /// preserving the magical essence while changing the form.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to convert
    /// * `from` - The source format
    /// * `to` - The target format
    ///
    /// # Returns
    ///
    /// The converted content as a string, or an error if conversion fails.
    ///
    /// # Errors
    ///
    /// Returns `Error::Config` if the conversion is not supported or fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let scribe = FormatterAgent::new();
    ///
    /// let json = r#"{"name": "test", "value": 42}"#;
    /// let yaml = scribe.convert(json, OutputFormat::Json, OutputFormat::Yaml)?;
    /// // Output:
    /// // name: test
    /// // value: 42
    /// ```
    pub fn convert(&self, content: &str, from: OutputFormat, to: OutputFormat) -> Result<String> {
        // If same format, return as-is
        if from == to {
            return Ok(content.to_string());
        }

        // Parse from source format
        let parsed = self.parse_format(content, &from)?;

        // Format to target format
        let formatted = self.render_format(&parsed, &to)?;

        Ok(formatted)
    }

    // =========================================================================
    // FORMAT-SPECIFIC FORMATTING
    // =========================================================================

    /// Format content as Markdown.
    fn format_as_markdown(&self, content: &str) -> String {
        let trimmed = content.trim();

        // If already looks like markdown, return as-is
        if self.looks_like_markdown(trimmed) {
            return trimmed.to_string();
        }

        // Convert plain text to basic markdown
        let lines: Vec<&str> = trimmed.lines().collect();

        if lines.len() == 1 {
            // Single line - check if it's a title-like text
            if trimmed.len() < 80 && !trimmed.contains('.') {
                return format!("# {}", trimmed);
            }
            return trimmed.to_string();
        }

        // Multiple lines - format with proper spacing
        lines.join("\n\n")
    }

    /// Format content as JSON.
    fn format_as_json(&self, content: &str) -> String {
        let trimmed = content.trim();

        // If already valid JSON, pretty-print it
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return serde_json::to_string_pretty(&value).unwrap_or_else(|_| trimmed.to_string());
        }

        // Convert to simple JSON structure
        let lines: Vec<&str> = trimmed.lines().map(|l| l.trim()).collect();

        if lines.len() == 1 {
            // Single value
            serde_json::json!({ "value": lines[0] }).to_string()
        } else {
            // Array of values
            serde_json::json!({ "items": lines }).to_string()
        }
    }

    /// Format content as YAML.
    fn format_as_yaml(&self, content: &str) -> String {
        let trimmed = content.trim();

        // If already valid YAML, return as-is
        if trimmed.starts_with("---") || self.looks_like_yaml(trimmed) {
            return trimmed.to_string();
        }

        // Try parsing as JSON first
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return serde_yaml::to_string(&value).unwrap_or_else(|_| trimmed.to_string());
        }

        // Convert lines to YAML list
        let lines: Vec<&str> = trimmed.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();

        if lines.len() == 1 {
            format!("value: {}", lines[0])
        } else {
            let items: Vec<String> = lines.iter().map(|l| format!("- {}", l)).collect();
            format!("items:\n{}", items.join("\n"))
        }
    }

    /// Format content as plain text.
    fn format_as_plain(&self, content: &str) -> String {
        let trimmed = content.trim();

        // Strip markdown formatting
        self.strip_markdown(trimmed)
    }

    /// Format content as code.
    fn format_as_code(&self, content: &str, language: &str) -> String {
        let trimmed = content.trim();

        // Wrap in code block if not already
        if trimmed.starts_with("```") {
            return trimmed.to_string();
        }

        format!("```{}\n{}\n```", language, trimmed)
    }

    /// Format content as a table.
    fn format_as_table(&self, content: &str) -> String {
        let trimmed = content.trim();

        // If already looks like a table, return as-is
        if trimmed.starts_with('|') {
            return trimmed.to_string();
        }

        // Try to parse as CSV-like data
        let lines: Vec<&str> = trimmed.lines().collect();

        if lines.is_empty() {
            return String::new();
        }

        // Detect separator (comma, tab, or pipe)
        let separator = if lines[0].contains('\t') {
            '\t'
        } else if lines[0].contains(',') {
            ','
        } else {
            ' '
        };

        let mut result = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            let cells: Vec<&str> = line.split(separator).map(|s| s.trim()).collect();
            let row = format!("| {} |", cells.join(" | "));
            result.push(row);

            // Add header separator after first row
            if i == 0 {
                let sep = cells.iter().map(|_| "---").collect::<Vec<_>>().join(" | ");
                result.push(format!("| {} |", sep));
            }
        }

        result.join("\n")
    }

    /// Format content as bullet list.
    fn format_as_bullet(&self, content: &str) -> String {
        let trimmed = content.trim();

        // If already a bullet list, return as-is
        if self.looks_like_bullet_list(trimmed) {
            return trimmed.to_string();
        }

        // Split by common separators
        let items: Vec<&str> = if trimmed.contains('\n') {
            trimmed.lines().collect()
        } else if trimmed.contains(',') {
            trimmed.split(',').collect()
        } else if trimmed.contains(';') {
            trimmed.split(';').collect()
        } else {
            vec![trimmed]
        };

        items
            .iter()
            .map(|item| item.trim())
            .filter(|item| !item.is_empty())
            .map(|item| format!("- {}", item))
            .collect::<Vec<_>>()
            .join("\n")
    }

    // =========================================================================
    // FORMAT DETECTION
    // =========================================================================

    /// Detect the format of the given content.
    ///
    /// Returns the detected format and confidence score.
    fn detect_format(&self, content: &str) -> (OutputFormat, f64) {
        let trimmed = content.trim();

        if trimmed.is_empty() {
            return (OutputFormat::PlainText, 1.0);
        }

        // Check each format with confidence scores
        let mut candidates: Vec<(OutputFormat, f64)> = Vec::new();

        // JSON detection
        if let Some(confidence) = self.detect_json(trimmed) {
            candidates.push((OutputFormat::Json, confidence));
        }

        // YAML detection
        if let Some(confidence) = self.detect_yaml(trimmed) {
            candidates.push((OutputFormat::Yaml, confidence));
        }

        // Markdown detection
        if let Some(confidence) = self.detect_markdown(trimmed) {
            candidates.push((OutputFormat::Markdown, confidence));
        }

        // Code detection
        if let Some((language, confidence)) = self.detect_code(trimmed) {
            candidates.push((OutputFormat::Code { language }, confidence));
        }

        // Table detection
        if let Some(confidence) = self.detect_table(trimmed) {
            candidates.push((OutputFormat::Table, confidence));
        }

        // Bullet list detection
        if let Some(confidence) = self.detect_bullet(trimmed) {
            candidates.push((OutputFormat::Bullet, confidence));
        }

        // Return highest confidence match
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates
            .into_iter()
            .next()
            .unwrap_or((OutputFormat::PlainText, 0.3))
    }

    /// Detect JSON format.
    fn detect_json(&self, content: &str) -> Option<f64> {
        // Check for JSON object or array pattern
        if self.patterns.json_object.is_match(content) || self.patterns.json_array.is_match(content)
        {
            // Validate it's actual JSON
            if serde_json::from_str::<serde_json::Value>(content).is_ok() {
                return Some(0.95);
            }
            return Some(0.6);
        }
        None
    }

    /// Detect YAML format.
    fn detect_yaml(&self, content: &str) -> Option<f64> {
        let lines: Vec<&str> = content.lines().collect();

        // Check for YAML header
        if !lines.is_empty() && self.patterns.yaml_header.is_match(lines[0]) {
            return Some(0.9);
        }

        // Count YAML-like key-value lines
        let yaml_lines = lines
            .iter()
            .filter(|l| self.patterns.yaml_key_value.is_match(l))
            .count();

        if yaml_lines > 0 {
            let ratio = yaml_lines as f64 / lines.len() as f64;
            if ratio > 0.5 {
                return Some(0.7 + ratio * 0.2);
            }
        }

        None
    }

    /// Detect Markdown format.
    fn detect_markdown(&self, content: &str) -> Option<f64> {
        let lines: Vec<&str> = content.lines().collect();
        let mut score: f64 = 0.0;
        let mut has_header = false;
        let mut has_emphasis = false;
        let mut has_code = false;
        let mut has_list = false;

        for line in &lines {
            if self.patterns.markdown_header.is_match(line) {
                score += 0.4;
                has_header = true;
            }
            if self.patterns.markdown_list.is_match(line) {
                score += 0.1;
                has_list = true;
            }
            if self.patterns.markdown_code_block.is_match(line) {
                score += 0.3;
                has_code = true;
            }
            if line.contains("**") || line.contains("__") {
                score += 0.2;
                has_emphasis = true;
            }
            if line.contains('`') && !line.contains("```") {
                score += 0.1;
            }
        }

        // Markdown requires at least headers OR multiple markdown features
        let feature_count = [has_header, has_emphasis, has_code, has_list]
            .iter()
            .filter(|&&x| x)
            .count();

        if has_header || feature_count >= 2 {
            // Cap score but ensure reasonable confidence
            Some(score.min(0.9).max(0.5))
        } else if feature_count == 1 && has_emphasis {
            // Just emphasis could be markdown
            Some(0.4)
        } else {
            None
        }
    }

    /// Detect code format and language.
    fn detect_code(&self, content: &str) -> Option<(String, f64)> {
        // Check for common code patterns
        if self.patterns.code_function.is_match(content) {
            let language = self.detect_code_language(content);
            return Some((language, 0.85));
        }

        if self.patterns.code_class.is_match(content) {
            let language = self.detect_code_language(content);
            return Some((language, 0.8));
        }

        // Check for code block markers
        if content.starts_with("```") {
            let first_line = content.lines().next().unwrap_or("");
            let language = first_line.strip_prefix("```").unwrap_or("").trim().to_string();
            let lang = if language.is_empty() {
                "text".to_string()
            } else {
                language
            };
            return Some((lang, 0.95));
        }

        None
    }

    /// Detect programming language from code content.
    fn detect_code_language(&self, content: &str) -> String {
        // Check for language-specific patterns
        if content.contains("fn ") && content.contains("->") {
            return "rust".to_string();
        }
        if content.contains("def ") && content.contains(':') {
            return "python".to_string();
        }
        if content.contains("function ") || content.contains("const ") || content.contains("let ")
        {
            return "javascript".to_string();
        }
        if content.contains("public ") || content.contains("private ")
        {
            if content.contains("class ") {
                return "java".to_string();
            }
        }
        if content.contains("func ") && content.contains("{") {
            return "go".to_string();
        }

        "text".to_string()
    }

    /// Detect table format.
    fn detect_table(&self, content: &str) -> Option<f64> {
        let lines: Vec<&str> = content.lines().collect();

        if lines.is_empty() {
            return None;
        }

        // Count table-like rows
        let table_rows = lines
            .iter()
            .filter(|l| self.patterns.table_row.is_match(l))
            .count();

        if table_rows > 0 {
            let ratio = table_rows as f64 / lines.len() as f64;
            if ratio > 0.5 {
                return Some(0.8 + ratio * 0.1);
            }
        }

        // Check for CSV-like content
        let commas = content.matches(',').count();
        let lines_count = lines.len();
        if lines_count > 1 && commas >= lines_count {
            return Some(0.6);
        }

        None
    }

    /// Detect bullet list format.
    fn detect_bullet(&self, content: &str) -> Option<f64> {
        let lines: Vec<&str> = content.lines().collect();

        if lines.is_empty() {
            return None;
        }

        let bullet_lines = lines
            .iter()
            .filter(|l| self.patterns.bullet_item.is_match(l))
            .count();

        if bullet_lines > 0 {
            let ratio = bullet_lines as f64 / lines.len() as f64;
            if ratio > 0.5 {
                return Some(0.7 + ratio * 0.2);
            }
        }

        None
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// Check if content looks like Markdown.
    fn looks_like_markdown(&self, content: &str) -> bool {
        let lines: Vec<&str> = content.lines().collect();
        lines.iter().any(|l| {
            self.patterns.markdown_header.is_match(l)
                || self.patterns.markdown_list.is_match(l)
                || self.patterns.markdown_code_block.is_match(l)
                || l.contains("**")
                || l.contains("__")
        })
    }

    /// Check if content looks like YAML.
    fn looks_like_yaml(&self, content: &str) -> bool {
        let lines: Vec<&str> = content.lines().collect();
        let yaml_count = lines
            .iter()
            .filter(|l| self.patterns.yaml_key_value.is_match(l))
            .count();
        yaml_count > 0 && yaml_count >= lines.len() / 2
    }

    /// Check if content looks like a bullet list.
    fn looks_like_bullet_list(&self, content: &str) -> bool {
        let lines: Vec<&str> = content.lines().collect();
        let bullet_count = lines
            .iter()
            .filter(|l| self.patterns.bullet_item.is_match(l))
            .count();
        bullet_count > 0 && bullet_count >= lines.len() / 2
    }

    /// Strip Markdown formatting from content.
    fn strip_markdown(&self, content: &str) -> String {
        let mut result = content.to_string();

        // Remove code blocks
        let code_block_re = regex::Regex::new(r"```[\s\S]*?```").unwrap();
        result = code_block_re.replace_all(&result, "").to_string();

        // Remove inline code
        let inline_code_re = regex::Regex::new(r"`[^`]+`").unwrap();
        result = inline_code_re.replace_all(&result, "").to_string();

        // Remove headers
        let header_re = regex::Regex::new(r"^#{1,6}\s*").unwrap();
        result = result
            .lines()
            .map(|line| header_re.replace(line, "").to_string())
            .collect::<Vec<_>>()
            .join("\n");

        // Remove bold
        let bold_re = regex::Regex::new(r"\*\*([^*]+)\*\*").unwrap();
        result = bold_re.replace_all(&result, "$1").to_string();

        // Remove italic
        let italic_re = regex::Regex::new(r"\*([^*]+)\*").unwrap();
        result = italic_re.replace_all(&result, "$1").to_string();

        // Remove list markers
        let list_re = regex::Regex::new(r"^[\s]*[-*+]\s").unwrap();
        result = result
            .lines()
            .map(|line| list_re.replace(line, "").to_string())
            .collect::<Vec<_>>()
            .join("\n");

        result.trim().to_string()
    }

    /// Parse content from a specific format into an intermediate representation.
    fn parse_format(&self, content: &str, format: &OutputFormat) -> Result<serde_json::Value> {
        match format {
            OutputFormat::Json => {
                serde_json::from_str(content).map_err(|e| Error::Config(e.to_string()))
            }
            OutputFormat::Yaml => {
                serde_yaml::from_str(content).map_err(|e| Error::Config(e.to_string()))
            }
            _ => {
                // For non-structured formats, wrap in a simple structure
                Ok(serde_json::json!({ "content": content }))
            }
        }
    }

    /// Render an intermediate representation to a specific format.
    fn render_format(&self, value: &serde_json::Value, format: &OutputFormat) -> Result<String> {
        match format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(value).map_err(|e| Error::Config(e.to_string()))
            }
            OutputFormat::Yaml => {
                serde_yaml::to_string(value).map_err(|e| Error::Config(e.to_string()))
            }
            OutputFormat::PlainText => {
                // Extract content or serialize
                if let Some(content) = value.get("content").and_then(|v| v.as_str()) {
                    Ok(content.to_string())
                } else {
                    Ok(value.to_string())
                }
            }
            _ => {
                // For other formats, use JSON as intermediate
                serde_json::to_string_pretty(value).map_err(|e| Error::Config(e.to_string()))
            }
        }
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom FormatterAgent configuration.
#[derive(Debug, Default)]
pub struct FormatterAgentBuilder {
    default_format: Option<OutputFormat>,
    detection_threshold: Option<f64>,
    custom_templates: HashMap<String, FormatTemplate>,
}

impl FormatterAgentBuilder {
    /// Set the default format for auto-detection fallback.
    pub fn default_format(mut self, format: OutputFormat) -> Self {
        self.default_format = Some(format);
        self
    }

    /// Set the minimum confidence threshold for auto-detection.
    ///
    /// # Panics
    ///
    /// Panics if value is not in range 0.0..=1.0
    pub fn detection_threshold(mut self, threshold: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "Threshold must be between 0.0 and 1.0"
        );
        self.detection_threshold = Some(threshold);
        self
    }

    /// Add a custom template.
    pub fn add_template(mut self, name: impl Into<String>, template: FormatTemplate) -> Self {
        self.custom_templates.insert(name.into(), template);
        self
    }

    /// Build the FormatterAgent.
    pub fn build(self) -> FormatterAgent {
        let mut agent = FormatterAgent::new();

        if let Some(format) = self.default_format {
            agent.default_format = format;
        }

        if let Some(threshold) = self.detection_threshold {
            agent.detection_threshold = threshold;
        }

        for (name, template) in self.custom_templates {
            agent.templates.insert(name, template);
        }

        agent
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Markdown.to_string(), "markdown");
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Yaml.to_string(), "yaml");
        assert_eq!(OutputFormat::PlainText.to_string(), "plain");
        assert_eq!(OutputFormat::Table.to_string(), "table");
        assert_eq!(OutputFormat::Bullet.to_string(), "bullet");
        assert_eq!(
            OutputFormat::Code {
                language: "rust".to_string()
            }
            .to_string(),
            "code:rust"
        );
    }

    #[test]
    fn test_output_format_from_str() {
        assert_eq!("markdown".parse::<OutputFormat>().unwrap(), OutputFormat::Markdown);
        assert_eq!("md".parse::<OutputFormat>().unwrap(), OutputFormat::Markdown);
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("yaml".parse::<OutputFormat>().unwrap(), OutputFormat::Yaml);
        assert_eq!("yml".parse::<OutputFormat>().unwrap(), OutputFormat::Yaml);
        assert_eq!("plain".parse::<OutputFormat>().unwrap(), OutputFormat::PlainText);
        assert_eq!("table".parse::<OutputFormat>().unwrap(), OutputFormat::Table);
        assert_eq!("bullet".parse::<OutputFormat>().unwrap(), OutputFormat::Bullet);
        assert_eq!(
            "code:python".parse::<OutputFormat>().unwrap(),
            OutputFormat::Code {
                language: "python".to_string()
            }
        );
    }

    #[test]
    fn test_output_format_from_str_invalid() {
        assert!("invalid".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_formatted_output_new() {
        let output = FormattedOutput::new("test content", OutputFormat::Markdown);
        assert_eq!(output.content, "test content");
        assert_eq!(output.format, OutputFormat::Markdown);
        assert_eq!(output.original_length, 12);
        assert!(!output.auto_detected);
    }

    #[test]
    fn test_formatted_output_helpers() {
        let output = FormattedOutput {
            content: "longer content".to_string(),
            format: OutputFormat::Markdown,
            original_length: 5,
            formatted_length: 14,
            auto_detected: false,
            detection_confidence: 1.0,
            warnings: vec!["warning".to_string()],
        };

        assert!(output.has_warnings());
        assert_eq!(output.size_delta(), 9);
        assert!(output.was_transformed());
    }

    #[test]
    fn test_formatter_agent_creation() {
        let agent = FormatterAgent::new();
        assert_eq!(agent.default_format(), &OutputFormat::PlainText);
        assert!((agent.detection_threshold() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_format_as_markdown() {
        let agent = FormatterAgent::new();
        let result = agent.format("Short Title", OutputFormat::Markdown);
        assert!(result.content.starts_with('#'));
    }

    #[test]
    fn test_format_as_json() {
        let agent = FormatterAgent::new();
        let result = agent.format("test value", OutputFormat::Json);
        assert!(result.content.contains("value"));
        assert!(serde_json::from_str::<serde_json::Value>(&result.content).is_ok());
    }

    #[test]
    fn test_format_as_yaml() {
        let agent = FormatterAgent::new();
        let result = agent.format("test value", OutputFormat::Yaml);
        assert!(result.content.contains("value:"));
    }

    #[test]
    fn test_format_as_bullet() {
        let agent = FormatterAgent::new();
        let result = agent.format("item1, item2, item3", OutputFormat::Bullet);
        assert!(result.content.contains("- item1"));
        assert!(result.content.contains("- item2"));
        assert!(result.content.contains("- item3"));
    }

    #[test]
    fn test_format_as_table() {
        let agent = FormatterAgent::new();
        let result = agent.format("a,b,c\n1,2,3", OutputFormat::Table);
        assert!(result.content.contains('|'));
        assert!(result.content.contains("---"));
    }

    #[test]
    fn test_format_as_code() {
        let agent = FormatterAgent::new();
        let result = agent.format(
            "fn main() {}",
            OutputFormat::Code {
                language: "rust".to_string(),
            },
        );
        assert!(result.content.starts_with("```rust"));
        assert!(result.content.ends_with("```"));
    }

    #[test]
    fn test_auto_format_json() {
        let agent = FormatterAgent::new();
        let json = r#"{"key": "value"}"#;
        let result = agent.auto_format(json);
        assert_eq!(result.format, OutputFormat::Json);
        assert!(result.auto_detected);
        assert!(result.detection_confidence > 0.5);
    }

    #[test]
    fn test_auto_format_yaml() {
        let agent = FormatterAgent::new();
        let yaml = "---\nname: test\nvalue: 42";
        let result = agent.auto_format(yaml);
        assert_eq!(result.format, OutputFormat::Yaml);
    }

    #[test]
    fn test_auto_format_markdown() {
        // Use a more markdown-rich content to ensure detection
        let agent = FormatterAgent::builder()
            .detection_threshold(0.3)  // Lower threshold for markdown detection
            .build();
        let md = "# Header\n\n## Subheader\n\nSome **bold** and *italic* text\n\n- List item\n- Another item";
        let result = agent.auto_format(md);
        assert_eq!(result.format, OutputFormat::Markdown);
    }

    #[test]
    fn test_auto_format_bullet() {
        let agent = FormatterAgent::new();
        let bullet = "- item 1\n- item 2\n- item 3";
        let result = agent.auto_format(bullet);
        assert_eq!(result.format, OutputFormat::Bullet);
    }

    #[test]
    fn test_convert_json_to_yaml() {
        let agent = FormatterAgent::new();
        let json = r#"{"name": "test", "value": 42}"#;
        let result = agent.convert(json, OutputFormat::Json, OutputFormat::Yaml);
        assert!(result.is_ok());
        let yaml = result.unwrap();
        assert!(yaml.contains("name:"));
        assert!(yaml.contains("value:"));
    }

    #[test]
    fn test_convert_yaml_to_json() {
        let agent = FormatterAgent::new();
        let yaml = "name: test\nvalue: 42";
        let result = agent.convert(yaml, OutputFormat::Yaml, OutputFormat::Json);
        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json.contains("\"name\""));
        assert!(json.contains("\"value\""));
    }

    #[test]
    fn test_convert_same_format() {
        let agent = FormatterAgent::new();
        let content = "test content";
        let result = agent.convert(content, OutputFormat::PlainText, OutputFormat::PlainText);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), content);
    }

    #[test]
    fn test_detect_code_rust() {
        let agent = FormatterAgent::new();
        let rust_code = "fn main() -> Result<()> { Ok(()) }";
        let result = agent.auto_format(rust_code);
        if let OutputFormat::Code { language } = result.format {
            assert_eq!(language, "rust");
        } else {
            panic!("Expected Code format");
        }
    }

    #[test]
    fn test_detect_code_python() {
        let agent = FormatterAgent::new();
        let python_code = "def main():\n    print('hello')";
        let result = agent.auto_format(python_code);
        if let OutputFormat::Code { language } = result.format {
            assert_eq!(language, "python");
        } else {
            panic!("Expected Code format");
        }
    }

    #[test]
    fn test_builder_pattern() {
        let agent = FormatterAgent::builder()
            .default_format(OutputFormat::Markdown)
            .detection_threshold(0.7)
            .build();

        assert_eq!(agent.default_format(), &OutputFormat::Markdown);
        assert!((agent.detection_threshold() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_format_template_markdown() {
        let template = FormatTemplate::markdown();
        assert_eq!(template.format, OutputFormat::Markdown);
        assert_eq!(template.separator, "\n\n");
    }

    #[test]
    fn test_format_template_json() {
        let template = FormatTemplate::json();
        assert_eq!(template.format, OutputFormat::Json);
        assert_eq!(template.header, Some("{".to_string()));
        assert_eq!(template.footer, Some("}".to_string()));
    }

    #[test]
    fn test_format_template_bullet() {
        let template = FormatTemplate::bullet();
        assert_eq!(template.format, OutputFormat::Bullet);
        assert_eq!(template.item_prefix, "- ");
    }

    #[test]
    fn test_strip_markdown() {
        let agent = FormatterAgent::new();
        let md = "# Header\n\n**Bold** and *italic* text";
        let result = agent.format(md, OutputFormat::PlainText);
        assert!(!result.content.contains('#'));
        assert!(!result.content.contains('*'));
    }

    #[test]
    fn test_empty_content() {
        let agent = FormatterAgent::new();
        let result = agent.auto_format("");
        assert_eq!(result.format, OutputFormat::PlainText);
        assert_eq!(result.detection_confidence, 1.0);
    }

    #[test]
    fn test_table_detection() {
        let agent = FormatterAgent::new();
        let table = "| Header 1 | Header 2 |\n| --- | --- |\n| Cell 1 | Cell 2 |";
        let result = agent.auto_format(table);
        assert_eq!(result.format, OutputFormat::Table);
    }

    #[test]
    fn test_format_preserves_already_formatted() {
        let agent = FormatterAgent::new();
        let json = r#"{"key": "value"}"#;
        let result = agent.format(json, OutputFormat::Json);
        // Should be pretty-printed
        assert!(serde_json::from_str::<serde_json::Value>(&result.content).is_ok());
    }

    #[test]
    fn test_multiline_bullet_format() {
        let agent = FormatterAgent::new();
        let content = "First item\nSecond item\nThird item";
        let result = agent.format(content, OutputFormat::Bullet);
        assert!(result.content.contains("- First item"));
        assert!(result.content.contains("- Second item"));
        assert!(result.content.contains("- Third item"));
    }

    #[test]
    fn test_code_block_preserved() {
        let agent = FormatterAgent::new();
        let code = "```rust\nfn main() {}\n```";
        let result = agent.format(
            code,
            OutputFormat::Code {
                language: "rust".to_string(),
            },
        );
        // Should preserve existing code block
        assert_eq!(result.content, code);
    }
}
