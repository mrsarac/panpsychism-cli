//! Templater Agent module for Project Panpsychism.
//!
//! 🎭 **The Pattern Master** — Shapes raw magic into precise forms.
//!
//! This module implements Agent 32: the Templater (Pattern Master), responsible for
//! managing and instantiating prompt templates with variable substitution. Like a master
//! craftsman who molds clay into perfect vessels, the Templater transforms abstract
//! patterns into concrete, usable prompts.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, The Pattern Master serves a crucial role:
//!
//! - **Template Scrolls** (PromptTemplate) contain the incantation patterns
//! - **Arcane Variables** (TemplateVariable) are the slots for power words
//! - **The Pattern Master** (TemplaterAgent) binds variables to create spells
//! - **Instantiated Spell** (InstantiatedPrompt) is the ready-to-cast incantation
//!
//! The Pattern Master can:
//! - Parse and validate template syntax
//! - Substitute variables with type checking
//! - Apply default values for optional parameters
//! - Validate required variables are provided
//! - Track instantiation warnings and metadata
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//! - **CONATUS**: Templates persist and evolve across sessions
//! - **NATURA**: Natural mapping between variables and their values
//! - **RATIO**: Logical validation of types and requirements
//! - **LAETITIA**: Joy through reusable, well-crafted templates
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::templater::{TemplaterAgent, PromptTemplate, TemplateVariable, VariableType};
//! use std::collections::HashMap;
//!
//! let mut template = PromptTemplate::new("greeting", "Hello, {{name}}! Welcome to {{project}}.");
//! template.add_variable(TemplateVariable::required("name", VariableType::String));
//! template.add_variable(TemplateVariable::with_default("project", VariableType::String, "Panpsychism"));
//!
//! let agent = TemplaterAgent::builder()
//!     .add_template(template)
//!     .build();
//!
//! let mut vars = HashMap::new();
//! vars.insert("name".to_string(), "Alice".to_string());
//!
//! let result = agent.instantiate("greeting", &vars).await?;
//! println!("{}", result.content); // "Hello, Alice! Welcome to Panpsychism."
//! ```

use crate::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// =============================================================================
// VARIABLE TYPE ENUM
// =============================================================================

/// The types of values that can be substituted into templates.
///
/// Each type defines the expected format and validation rules for
/// template variables, ensuring type safety in template instantiation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum VariableType {
    /// A text string value.
    ///
    /// Accepts any valid UTF-8 string. This is the most flexible type
    /// and is the default for untyped variables.
    #[default]
    String,

    /// A numeric value (integer or floating point).
    ///
    /// Validates that the value can be parsed as a number.
    /// Accepts formats like "42", "-17", "3.14", "-0.5".
    Number,

    /// A boolean value (true/false).
    ///
    /// Accepts: "true", "false", "yes", "no", "1", "0".
    Boolean,

    /// A list of values.
    ///
    /// Values should be comma-separated or JSON array format.
    /// Example: "a, b, c" or '["a", "b", "c"]'
    List,

    /// A structured object (JSON format).
    ///
    /// Validates that the value is valid JSON.
    Object,

    /// An enumeration with predefined allowed values.
    ///
    /// The allowed values are specified in the `allowed_values` field.
    Enum {
        /// The set of allowed values for this enum.
        allowed_values: Vec<String>,
    },
}

impl VariableType {
    /// Create an enum type with allowed values.
    pub fn enumeration(values: Vec<impl Into<String>>) -> Self {
        Self::Enum {
            allowed_values: values.into_iter().map(|v| v.into()).collect(),
        }
    }

    /// Validate that a value matches this type.
    ///
    /// Returns `Ok(())` if valid, or an error message if invalid.
    pub fn validate(&self, value: &str) -> std::result::Result<(), String> {
        match self {
            Self::String => Ok(()), // Any string is valid
            Self::Number => {
                if value.parse::<f64>().is_ok() {
                    Ok(())
                } else {
                    Err(format!("'{}' is not a valid number", value))
                }
            }
            Self::Boolean => {
                let lower = value.to_lowercase();
                if ["true", "false", "yes", "no", "1", "0"].contains(&lower.as_str()) {
                    Ok(())
                } else {
                    Err(format!(
                        "'{}' is not a valid boolean (use true/false, yes/no, or 1/0)",
                        value
                    ))
                }
            }
            Self::List => {
                // Accept comma-separated or JSON array
                if value.starts_with('[') {
                    serde_json::from_str::<Vec<serde_json::Value>>(value)
                        .map(|_| ())
                        .map_err(|e| format!("Invalid list format: {}", e))
                } else {
                    Ok(()) // Comma-separated is always valid
                }
            }
            Self::Object => serde_json::from_str::<serde_json::Value>(value)
                .map(|_| ())
                .map_err(|e| format!("Invalid JSON object: {}", e)),
            Self::Enum { allowed_values } => {
                if allowed_values.contains(&value.to_string()) {
                    Ok(())
                } else {
                    Err(format!(
                        "'{}' is not a valid option. Allowed values: {}",
                        value,
                        allowed_values.join(", ")
                    ))
                }
            }
        }
    }

    /// Get a human-readable description of this type.
    pub fn description(&self) -> &'static str {
        match self {
            Self::String => "text string",
            Self::Number => "numeric value",
            Self::Boolean => "boolean (true/false)",
            Self::List => "list of values",
            Self::Object => "JSON object",
            Self::Enum { .. } => "enumeration",
        }
    }
}

impl std::fmt::Display for VariableType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Number => write!(f, "number"),
            Self::Boolean => write!(f, "boolean"),
            Self::List => write!(f, "list"),
            Self::Object => write!(f, "object"),
            Self::Enum { allowed_values } => {
                write!(f, "enum[{}]", allowed_values.join("|"))
            }
        }
    }
}

impl std::str::FromStr for VariableType {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let lower = s.to_lowercase();

        // Check for enum with values: "enum[a|b|c]"
        if lower.starts_with("enum[") && lower.ends_with(']') {
            let inner = &lower[5..lower.len() - 1];
            let values: Vec<String> = inner.split('|').map(|v| v.trim().to_string()).collect();
            if values.is_empty() || values.iter().all(|v| v.is_empty()) {
                return Err(Error::Config("Enum type requires at least one value".into()));
            }
            return Ok(Self::Enum {
                allowed_values: values,
            });
        }

        match lower.as_str() {
            "string" | "str" | "text" => Ok(Self::String),
            "number" | "num" | "int" | "float" | "integer" => Ok(Self::Number),
            "boolean" | "bool" => Ok(Self::Boolean),
            "list" | "array" => Ok(Self::List),
            "object" | "json" | "dict" | "map" => Ok(Self::Object),
            _ => Err(Error::Config(format!(
                "Unknown variable type: '{}'. Valid types: string, number, boolean, list, object, enum[a|b|c]",
                s
            ))),
        }
    }
}

// =============================================================================
// TEMPLATE VARIABLE
// =============================================================================

/// A variable definition within a template.
///
/// Variables are the dynamic parts of templates that get substituted
/// during instantiation. Each variable has a name, type, and can be
/// required or optional (with a default value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    /// The name of the variable (used in {{name}} syntax).
    pub name: String,

    /// The expected type of the variable value.
    pub var_type: VariableType,

    /// Whether this variable is required.
    pub required: bool,

    /// Default value if not provided (for optional variables).
    pub default: Option<String>,

    /// Validation pattern (regex) for additional validation.
    pub validation: Option<String>,

    /// Human-readable description of this variable.
    pub description: Option<String>,
}

impl TemplateVariable {
    /// Create a new required string variable.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            var_type: VariableType::String,
            required: true,
            default: None,
            validation: None,
            description: None,
        }
    }

    /// Create a required variable with a specific type.
    pub fn required(name: impl Into<String>, var_type: VariableType) -> Self {
        Self {
            name: name.into(),
            var_type,
            required: true,
            default: None,
            validation: None,
            description: None,
        }
    }

    /// Create an optional variable with a default value.
    pub fn with_default(
        name: impl Into<String>,
        var_type: VariableType,
        default: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            var_type,
            required: false,
            default: Some(default.into()),
            validation: None,
            description: None,
        }
    }

    /// Create an optional variable without a default.
    pub fn optional(name: impl Into<String>, var_type: VariableType) -> Self {
        Self {
            name: name.into(),
            var_type,
            required: false,
            default: None,
            validation: None,
            description: None,
        }
    }

    /// Set a validation regex pattern.
    pub fn with_validation(mut self, pattern: impl Into<String>) -> Self {
        self.validation = Some(pattern.into());
        self
    }

    /// Set a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Validate a value against this variable's type and constraints.
    pub fn validate_value(&self, value: &str) -> std::result::Result<(), String> {
        // Type validation
        self.var_type.validate(value)?;

        // Regex validation if specified
        if let Some(ref pattern) = self.validation {
            let re = Regex::new(pattern)
                .map_err(|e| format!("Invalid validation pattern '{}': {}", pattern, e))?;
            if !re.is_match(value) {
                return Err(format!(
                    "Value '{}' does not match validation pattern '{}'",
                    value, pattern
                ));
            }
        }

        Ok(())
    }

    /// Get the effective value for this variable.
    ///
    /// Returns the provided value, the default, or None if neither exists.
    pub fn get_effective_value(&self, provided: Option<&String>) -> Option<String> {
        provided.cloned().or_else(|| self.default.clone())
    }
}

// =============================================================================
// PROMPT TEMPLATE
// =============================================================================

/// A reusable prompt template with variable placeholders.
///
/// Templates are the patterns from which prompts are instantiated.
/// They contain static content with variable placeholders in the
/// format `{{variable_name}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Unique identifier for this template.
    pub id: String,

    /// Human-readable name for the template.
    pub name: String,

    /// The template content with {{variable}} placeholders.
    pub content: String,

    /// Variable definitions for this template.
    pub variables: HashMap<String, TemplateVariable>,

    /// Version of this template (for tracking changes).
    pub version: String,

    /// Category for organization (e.g., "coding", "writing", "analysis").
    pub category: Option<String>,

    /// Tags for searchability.
    pub tags: Vec<String>,

    /// Description of what this template does.
    pub description: Option<String>,

    /// Author or creator of the template.
    pub author: Option<String>,

    /// Creation timestamp (ISO 8601 format).
    pub created_at: Option<String>,

    /// Last update timestamp.
    pub updated_at: Option<String>,
}

impl PromptTemplate {
    /// Create a new template with an ID and content.
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        let id_str = id.into();
        Self {
            id: id_str.clone(),
            name: id_str,
            content: content.into(),
            variables: HashMap::new(),
            version: "1.0.0".to_string(),
            category: None,
            tags: Vec::new(),
            description: None,
            author: None,
            created_at: None,
            updated_at: None,
        }
    }

    /// Set the template name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the template version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the template category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Add a variable definition.
    pub fn add_variable(&mut self, variable: TemplateVariable) {
        self.variables.insert(variable.name.clone(), variable);
    }

    /// Add a variable definition (builder pattern).
    pub fn with_variable(mut self, variable: TemplateVariable) -> Self {
        self.add_variable(variable);
        self
    }

    /// Extract variable names from the template content.
    ///
    /// Scans the content for `{{variable_name}}` patterns and returns
    /// all unique variable names found.
    pub fn extract_variable_names(&self) -> Vec<String> {
        let re = Regex::new(r"\{\{(\w+)\}\}").expect("Invalid regex");
        let mut names: Vec<String> = re
            .captures_iter(&self.content)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Check if all variables in the content have definitions.
    ///
    /// Returns a list of undefined variable names.
    pub fn undefined_variables(&self) -> Vec<String> {
        self.extract_variable_names()
            .into_iter()
            .filter(|name| !self.variables.contains_key(name))
            .collect()
    }

    /// Check if all required variables have definitions.
    pub fn required_variables(&self) -> Vec<&TemplateVariable> {
        self.variables.values().filter(|v| v.required).collect()
    }

    /// Check if all optional variables have definitions.
    pub fn optional_variables(&self) -> Vec<&TemplateVariable> {
        self.variables.values().filter(|v| !v.required).collect()
    }

    /// Validate the template structure.
    ///
    /// Checks that all variables in content have definitions and
    /// that the template syntax is valid.
    pub fn validate(&self) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check for undefined variables
        let undefined = self.undefined_variables();
        if !undefined.is_empty() {
            errors.push(format!(
                "Undefined variables in template: {}",
                undefined.join(", ")
            ));
        }

        // Check for empty content
        if self.content.trim().is_empty() {
            errors.push("Template content is empty".to_string());
        }

        // Check for malformed placeholders:
        // - Unclosed: {{ without matching }}
        // - Single brace: { or } alone (not {{ or }})
        // The valid pattern is {{word}} - we look for deviations

        // Count opening and closing double braces
        let opens = self.content.matches("{{").count();
        let closes = self.content.matches("}}").count();
        if opens != closes {
            errors.push("Template contains unbalanced placeholder braces".to_string());
        }

        // Check for single braces that aren't part of double braces
        // This is tricky - we'll just ensure every {{ has a matching }}
        // A more sophisticated check would use a state machine
        let single_open = self.content.matches('{').count();
        let single_close = self.content.matches('}').count();
        if single_open != opens * 2 || single_close != closes * 2 {
            errors.push("Template contains single braces (use {{ and }} for placeholders)".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// =============================================================================
// INSTANTIATED PROMPT
// =============================================================================

/// The result of template instantiation.
///
/// Contains the final prompt content with all variables substituted,
/// along with metadata about the instantiation process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstantiatedPrompt {
    /// The final prompt content with all variables substituted.
    pub content: String,

    /// The ID of the template that was instantiated.
    pub template_id: String,

    /// The template version that was used.
    pub template_version: String,

    /// Variables that were used in the instantiation.
    pub variables_used: HashMap<String, String>,

    /// Warnings generated during instantiation.
    pub warnings: Vec<String>,

    /// Processing time in milliseconds.
    pub processing_time_ms: u64,

    /// Whether any default values were applied.
    pub defaults_applied: Vec<String>,

    /// Variables that were provided but not used.
    pub unused_variables: Vec<String>,
}

impl InstantiatedPrompt {
    /// Check if instantiation generated any warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Check if any default values were applied.
    pub fn used_defaults(&self) -> bool {
        !self.defaults_applied.is_empty()
    }

    /// Check if any provided variables were unused.
    pub fn has_unused_variables(&self) -> bool {
        !self.unused_variables.is_empty()
    }

    /// Get the number of variables that were substituted.
    pub fn variable_count(&self) -> usize {
        self.variables_used.len()
    }
}

// =============================================================================
// TEMPLATE REGISTRY
// =============================================================================

/// A registry for managing multiple templates.
///
/// The registry provides storage, retrieval, and organization of templates.
/// It supports categorization, tagging, and versioning.
#[derive(Debug, Clone, Default)]
pub struct TemplateRegistry {
    /// All registered templates, indexed by ID.
    templates: HashMap<String, PromptTemplate>,

    /// Templates indexed by category.
    by_category: HashMap<String, Vec<String>>,

    /// Templates indexed by tag.
    by_tag: HashMap<String, Vec<String>>,
}

impl TemplateRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template in the registry.
    pub fn register(&mut self, template: PromptTemplate) {
        let id = template.id.clone();

        // Index by category
        if let Some(ref category) = template.category {
            self.by_category
                .entry(category.clone())
                .or_default()
                .push(id.clone());
        }

        // Index by tags
        for tag in &template.tags {
            self.by_tag
                .entry(tag.clone())
                .or_default()
                .push(id.clone());
        }

        self.templates.insert(id, template);
    }

    /// Get a template by ID.
    pub fn get(&self, id: &str) -> Option<&PromptTemplate> {
        self.templates.get(id)
    }

    /// Get all template IDs.
    pub fn template_ids(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    /// Get all templates.
    pub fn all_templates(&self) -> Vec<&PromptTemplate> {
        self.templates.values().collect()
    }

    /// Get templates by category.
    pub fn by_category(&self, category: &str) -> Vec<&PromptTemplate> {
        self.by_category
            .get(category)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.templates.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get templates by tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&PromptTemplate> {
        self.by_tag
            .get(tag)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.templates.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all categories.
    pub fn categories(&self) -> Vec<&str> {
        self.by_category.keys().map(|s| s.as_str()).collect()
    }

    /// Get all tags.
    pub fn tags(&self) -> Vec<&str> {
        self.by_tag.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of templates in the registry.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Remove a template by ID.
    pub fn remove(&mut self, id: &str) -> Option<PromptTemplate> {
        if let Some(template) = self.templates.remove(id) {
            // Clean up category index
            if let Some(ref category) = template.category {
                if let Some(ids) = self.by_category.get_mut(category) {
                    ids.retain(|i| i != id);
                }
            }

            // Clean up tag index
            for tag in &template.tags {
                if let Some(ids) = self.by_tag.get_mut(tag) {
                    ids.retain(|i| i != id);
                }
            }

            Some(template)
        } else {
            None
        }
    }

    /// Check if a template exists.
    pub fn contains(&self, id: &str) -> bool {
        self.templates.contains_key(id)
    }

    /// Search templates by name or description.
    pub fn search(&self, query: &str) -> Vec<&PromptTemplate> {
        let query_lower = query.to_lowercase();
        self.templates
            .values()
            .filter(|t| {
                t.name.to_lowercase().contains(&query_lower)
                    || t.id.to_lowercase().contains(&query_lower)
                    || t.description
                        .as_ref()
                        .map(|d| d.to_lowercase().contains(&query_lower))
                        .unwrap_or(false)
            })
            .collect()
    }
}

// =============================================================================
// TEMPLATER CONFIG
// =============================================================================

/// Configuration for the Templater Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplaterConfig {
    /// Whether to allow undefined variables in templates.
    ///
    /// If false, instantiation will fail if a variable in the template
    /// has no definition. If true, undefined variables are left as-is.
    #[serde(default)]
    pub allow_undefined: bool,

    /// Whether to validate variable types strictly.
    ///
    /// If true, type mismatches will cause instantiation to fail.
    /// If false, type mismatches generate warnings but succeed.
    #[serde(default = "default_true")]
    pub strict_types: bool,

    /// Whether to trim whitespace from variable values.
    #[serde(default = "default_true")]
    pub trim_values: bool,

    /// Whether to track unused provided variables.
    #[serde(default = "default_true")]
    pub track_unused: bool,

    /// Maximum number of nested template references.
    ///
    /// Prevents infinite recursion in nested templates.
    #[serde(default = "default_max_depth")]
    pub max_recursion_depth: usize,

    /// Custom variable pattern (default: `{{name}}`).
    ///
    /// The pattern must contain a capture group for the variable name.
    #[serde(default)]
    pub variable_pattern: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_max_depth() -> usize {
    10
}

impl Default for TemplaterConfig {
    fn default() -> Self {
        Self {
            allow_undefined: false,
            strict_types: true,
            trim_values: true,
            track_unused: true,
            max_recursion_depth: 10,
            variable_pattern: None,
        }
    }
}

impl TemplaterConfig {
    /// Create a strict configuration (fail on any issue).
    pub fn strict() -> Self {
        Self {
            allow_undefined: false,
            strict_types: true,
            trim_values: true,
            track_unused: true,
            max_recursion_depth: 10,
            variable_pattern: None,
        }
    }

    /// Create a lenient configuration (warn but continue).
    pub fn lenient() -> Self {
        Self {
            allow_undefined: true,
            strict_types: false,
            trim_values: true,
            track_unused: true,
            max_recursion_depth: 10,
            variable_pattern: None,
        }
    }
}

// =============================================================================
// TEMPLATER AGENT (THE PATTERN MASTER)
// =============================================================================

/// The Templater Agent — The Pattern Master of the Sorcerer's Tower.
///
/// Agent 32 manages and instantiates prompt templates with variable substitution.
/// Like a master craftsman who shapes raw materials into refined works,
/// the Pattern Master transforms template patterns into ready-to-use prompts.
///
/// # Responsibilities
///
/// 1. **Manage Templates**: Store, retrieve, and organize template definitions
/// 2. **Validate Variables**: Ensure type safety and required values
/// 3. **Substitute Values**: Replace placeholders with actual values
/// 4. **Apply Defaults**: Use default values for optional parameters
/// 5. **Track Warnings**: Report issues during instantiation
///
/// # Example
///
/// ```rust,ignore
/// let agent = TemplaterAgent::builder()
///     .config(TemplaterConfig::default())
///     .add_template(template)
///     .build();
///
/// let mut vars = HashMap::new();
/// vars.insert("name".to_string(), "World".to_string());
///
/// let result = agent.instantiate("hello", &vars).await?;
/// ```
#[derive(Debug, Clone)]
pub struct TemplaterAgent {
    /// Template registry.
    registry: TemplateRegistry,

    /// Agent configuration.
    config: TemplaterConfig,

    /// Compiled variable pattern regex.
    variable_regex: Arc<Regex>,
}

impl Default for TemplaterAgent {
    fn default() -> Self {
        Self::new(TemplaterConfig::default())
    }
}

impl TemplaterAgent {
    /// Create a new Templater Agent with default configuration.
    pub fn new(config: TemplaterConfig) -> Self {
        let pattern = config
            .variable_pattern
            .as_deref()
            .unwrap_or(r"\{\{(\w+)\}\}");
        let regex = Regex::new(pattern).expect("Invalid variable pattern");

        Self {
            registry: TemplateRegistry::new(),
            config,
            variable_regex: Arc::new(regex),
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> TemplaterAgentBuilder {
        TemplaterAgentBuilder::default()
    }

    /// Get the template registry.
    pub fn registry(&self) -> &TemplateRegistry {
        &self.registry
    }

    /// Get the configuration.
    pub fn config(&self) -> &TemplaterConfig {
        &self.config
    }

    /// Register a template.
    pub fn register_template(&mut self, template: PromptTemplate) {
        self.registry.register(template);
    }

    /// Get a template by ID.
    pub fn get_template(&self, id: &str) -> Option<&PromptTemplate> {
        self.registry.get(id)
    }

    // =========================================================================
    // CORE INSTANTIATION
    // =========================================================================

    /// Instantiate a template with the given variables.
    ///
    /// This is the main entry point for template instantiation. It substitutes
    /// all variables in the template with the provided values, applies defaults,
    /// and validates types according to the configuration.
    ///
    /// # Arguments
    ///
    /// * `template_id` - The ID of the template to instantiate
    /// * `variables` - Map of variable names to their values
    ///
    /// # Returns
    ///
    /// An `InstantiatedPrompt` containing the final content and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The template is not found
    /// - Required variables are missing
    /// - Type validation fails (in strict mode)
    pub async fn instantiate(
        &self,
        template_id: &str,
        variables: &HashMap<String, String>,
    ) -> Result<InstantiatedPrompt> {
        let start = Instant::now();

        // Get the template
        let template = self.registry.get(template_id).ok_or_else(|| {
            Error::SynthesisTemplateNotFound {
                template: template_id.to_string(),
            }
        })?;

        self.instantiate_template(template, variables, start).await
    }

    /// Instantiate a template directly (without looking up by ID).
    pub async fn instantiate_template(
        &self,
        template: &PromptTemplate,
        variables: &HashMap<String, String>,
        start: Instant,
    ) -> Result<InstantiatedPrompt> {
        let mut warnings = Vec::new();
        let mut defaults_applied = Vec::new();
        let mut variables_used = HashMap::new();

        // Process each variable in the template content
        let content_vars = template.extract_variable_names();
        let mut final_content = template.content.clone();

        for var_name in &content_vars {
            let var_def = template.variables.get(var_name);
            let provided_value = variables.get(var_name);

            // Determine the value to use
            let value = if let Some(def) = var_def {
                match def.get_effective_value(provided_value) {
                    Some(mut v) => {
                        // Trim if configured
                        if self.config.trim_values {
                            v = v.trim().to_string();
                        }

                        // Validate type
                        if let Err(e) = def.validate_value(&v) {
                            if self.config.strict_types {
                                return Err(Error::SynthesisInvalidParams { details: e });
                            } else {
                                warnings.push(format!("Variable '{}': {}", var_name, e));
                            }
                        }

                        // Track if default was used
                        if provided_value.is_none() && def.default.is_some() {
                            defaults_applied.push(var_name.clone());
                        }

                        Some(v)
                    }
                    None => {
                        if def.required {
                            return Err(Error::SynthesisInvalidParams {
                                details: format!("Required variable '{}' is missing", var_name),
                            });
                        }
                        None
                    }
                }
            } else if self.config.allow_undefined {
                // No definition, use provided value or leave placeholder
                provided_value.cloned()
            } else {
                return Err(Error::SynthesisInvalidParams {
                    details: format!("Variable '{}' has no definition", var_name),
                });
            };

            // Substitute the variable
            if let Some(val) = value {
                let placeholder = format!("{{{{{}}}}}", var_name);
                final_content = final_content.replace(&placeholder, &val);
                variables_used.insert(var_name.clone(), val);
            }
        }

        // Track unused provided variables
        let unused_variables: Vec<String> = if self.config.track_unused {
            variables
                .keys()
                .filter(|k| !content_vars.contains(k))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        if !unused_variables.is_empty() {
            warnings.push(format!(
                "Unused variables provided: {}",
                unused_variables.join(", ")
            ));
        }

        let processing_time_ms = start.elapsed().as_millis() as u64;

        Ok(InstantiatedPrompt {
            content: final_content,
            template_id: template.id.clone(),
            template_version: template.version.clone(),
            variables_used,
            warnings,
            processing_time_ms,
            defaults_applied,
            unused_variables,
        })
    }

    /// Instantiate a template by content string (ad-hoc template).
    ///
    /// Creates a temporary template from the content and instantiates it.
    pub async fn instantiate_content(
        &self,
        content: &str,
        variables: &HashMap<String, String>,
    ) -> Result<InstantiatedPrompt> {
        let start = Instant::now();

        // Create a temporary template with auto-detected variables
        let mut template = PromptTemplate::new("_adhoc_", content);

        // Auto-create variable definitions for all detected variables
        let var_names = template.extract_variable_names();
        for name in var_names {
            template.add_variable(TemplateVariable::optional(&name, VariableType::String));
        }

        self.instantiate_template(&template, variables, start).await
    }

    /// Preview a template instantiation without validation.
    ///
    /// Useful for showing what the result would look like.
    pub fn preview(&self, template_id: &str, variables: &HashMap<String, String>) -> Result<String> {
        let template = self.registry.get(template_id).ok_or_else(|| {
            Error::SynthesisTemplateNotFound {
                template: template_id.to_string(),
            }
        })?;

        let mut content = template.content.clone();
        for (name, value) in variables {
            let placeholder = format!("{{{{{}}}}}", name);
            content = content.replace(&placeholder, value);
        }

        Ok(content)
    }

    /// Validate a template.
    pub fn validate_template(&self, template_id: &str) -> Result<()> {
        let template = self.registry.get(template_id).ok_or_else(|| {
            Error::SynthesisTemplateNotFound {
                template: template_id.to_string(),
            }
        })?;

        template
            .validate()
            .map_err(|errors| Error::SynthesisInvalidParams {
                details: errors.join("; "),
            })
    }

    /// List all template IDs.
    pub fn list_templates(&self) -> Vec<&str> {
        self.registry.template_ids()
    }

    /// Search templates by query.
    pub fn search_templates(&self, query: &str) -> Vec<&PromptTemplate> {
        self.registry.search(query)
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom TemplaterAgent configuration.
#[derive(Debug, Default)]
pub struct TemplaterAgentBuilder {
    config: Option<TemplaterConfig>,
    templates: Vec<PromptTemplate>,
}

impl TemplaterAgentBuilder {
    /// Set the configuration.
    pub fn config(mut self, config: TemplaterConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add a template.
    pub fn add_template(mut self, template: PromptTemplate) -> Self {
        self.templates.push(template);
        self
    }

    /// Enable strict mode.
    pub fn strict(mut self) -> Self {
        self.config = Some(TemplaterConfig::strict());
        self
    }

    /// Enable lenient mode.
    pub fn lenient(mut self) -> Self {
        self.config = Some(TemplaterConfig::lenient());
        self
    }

    /// Build the TemplaterAgent.
    pub fn build(self) -> TemplaterAgent {
        let mut agent = TemplaterAgent::new(self.config.unwrap_or_default());

        for template in self.templates {
            agent.register_template(template);
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

    // =========================================================================
    // VariableType Tests
    // =========================================================================

    #[test]
    fn test_variable_type_default() {
        assert_eq!(VariableType::default(), VariableType::String);
    }

    #[test]
    fn test_variable_type_display() {
        assert_eq!(VariableType::String.to_string(), "string");
        assert_eq!(VariableType::Number.to_string(), "number");
        assert_eq!(VariableType::Boolean.to_string(), "boolean");
        assert_eq!(VariableType::List.to_string(), "list");
        assert_eq!(VariableType::Object.to_string(), "object");
        assert_eq!(
            VariableType::enumeration(vec!["a", "b"]).to_string(),
            "enum[a|b]"
        );
    }

    #[test]
    fn test_variable_type_from_str() {
        assert_eq!("string".parse::<VariableType>().unwrap(), VariableType::String);
        assert_eq!("str".parse::<VariableType>().unwrap(), VariableType::String);
        assert_eq!("text".parse::<VariableType>().unwrap(), VariableType::String);
        assert_eq!("number".parse::<VariableType>().unwrap(), VariableType::Number);
        assert_eq!("int".parse::<VariableType>().unwrap(), VariableType::Number);
        assert_eq!("float".parse::<VariableType>().unwrap(), VariableType::Number);
        assert_eq!("boolean".parse::<VariableType>().unwrap(), VariableType::Boolean);
        assert_eq!("bool".parse::<VariableType>().unwrap(), VariableType::Boolean);
        assert_eq!("list".parse::<VariableType>().unwrap(), VariableType::List);
        assert_eq!("array".parse::<VariableType>().unwrap(), VariableType::List);
        assert_eq!("object".parse::<VariableType>().unwrap(), VariableType::Object);
        assert_eq!("json".parse::<VariableType>().unwrap(), VariableType::Object);
    }

    #[test]
    fn test_variable_type_from_str_enum() {
        let result = "enum[a|b|c]".parse::<VariableType>().unwrap();
        if let VariableType::Enum { allowed_values } = result {
            assert_eq!(allowed_values, vec!["a", "b", "c"]);
        } else {
            panic!("Expected Enum variant");
        }
    }

    #[test]
    fn test_variable_type_from_str_invalid() {
        assert!("invalid".parse::<VariableType>().is_err());
    }

    #[test]
    fn test_variable_type_validate_string() {
        let vt = VariableType::String;
        assert!(vt.validate("anything").is_ok());
        assert!(vt.validate("").is_ok());
        assert!(vt.validate("123").is_ok());
    }

    #[test]
    fn test_variable_type_validate_number() {
        let vt = VariableType::Number;
        assert!(vt.validate("42").is_ok());
        assert!(vt.validate("-17").is_ok());
        assert!(vt.validate("3.14").is_ok());
        assert!(vt.validate("-0.5").is_ok());
        assert!(vt.validate("not a number").is_err());
    }

    #[test]
    fn test_variable_type_validate_boolean() {
        let vt = VariableType::Boolean;
        assert!(vt.validate("true").is_ok());
        assert!(vt.validate("false").is_ok());
        assert!(vt.validate("TRUE").is_ok());
        assert!(vt.validate("yes").is_ok());
        assert!(vt.validate("no").is_ok());
        assert!(vt.validate("1").is_ok());
        assert!(vt.validate("0").is_ok());
        assert!(vt.validate("maybe").is_err());
    }

    #[test]
    fn test_variable_type_validate_list() {
        let vt = VariableType::List;
        assert!(vt.validate("a, b, c").is_ok());
        assert!(vt.validate(r#"["a", "b", "c"]"#).is_ok());
        assert!(vt.validate("[1, 2, 3]").is_ok());
        assert!(vt.validate("[invalid json").is_err());
    }

    #[test]
    fn test_variable_type_validate_object() {
        let vt = VariableType::Object;
        assert!(vt.validate(r#"{"key": "value"}"#).is_ok());
        assert!(vt.validate(r#"{"nested": {"a": 1}}"#).is_ok());
        assert!(vt.validate("not json").is_err());
    }

    #[test]
    fn test_variable_type_validate_enum() {
        let vt = VariableType::enumeration(vec!["red", "green", "blue"]);
        assert!(vt.validate("red").is_ok());
        assert!(vt.validate("green").is_ok());
        assert!(vt.validate("blue").is_ok());
        assert!(vt.validate("yellow").is_err());
    }

    #[test]
    fn test_variable_type_description() {
        assert_eq!(VariableType::String.description(), "text string");
        assert_eq!(VariableType::Number.description(), "numeric value");
        assert_eq!(VariableType::Boolean.description(), "boolean (true/false)");
        assert_eq!(VariableType::List.description(), "list of values");
        assert_eq!(VariableType::Object.description(), "JSON object");
        assert_eq!(VariableType::enumeration(vec!["a"]).description(), "enumeration");
    }

    // =========================================================================
    // TemplateVariable Tests
    // =========================================================================

    #[test]
    fn test_template_variable_new() {
        let var = TemplateVariable::new("name");
        assert_eq!(var.name, "name");
        assert_eq!(var.var_type, VariableType::String);
        assert!(var.required);
        assert!(var.default.is_none());
    }

    #[test]
    fn test_template_variable_required() {
        let var = TemplateVariable::required("count", VariableType::Number);
        assert_eq!(var.name, "count");
        assert_eq!(var.var_type, VariableType::Number);
        assert!(var.required);
    }

    #[test]
    fn test_template_variable_with_default() {
        let var = TemplateVariable::with_default("lang", VariableType::String, "en");
        assert_eq!(var.name, "lang");
        assert!(!var.required);
        assert_eq!(var.default, Some("en".to_string()));
    }

    #[test]
    fn test_template_variable_optional() {
        let var = TemplateVariable::optional("suffix", VariableType::String);
        assert!(!var.required);
        assert!(var.default.is_none());
    }

    #[test]
    fn test_template_variable_with_validation() {
        let var = TemplateVariable::new("email").with_validation(r"^[\w.-]+@[\w.-]+\.\w+$");
        assert!(var.validation.is_some());
    }

    #[test]
    fn test_template_variable_with_description() {
        let var = TemplateVariable::new("name").with_description("The user's name");
        assert_eq!(var.description, Some("The user's name".to_string()));
    }

    #[test]
    fn test_template_variable_validate_value() {
        let var = TemplateVariable::required("count", VariableType::Number);
        assert!(var.validate_value("42").is_ok());
        assert!(var.validate_value("not a number").is_err());
    }

    #[test]
    fn test_template_variable_validate_with_regex() {
        let var = TemplateVariable::new("code").with_validation(r"^[A-Z]{3}$");
        assert!(var.validate_value("ABC").is_ok());
        assert!(var.validate_value("AB").is_err());
        assert!(var.validate_value("abc").is_err());
    }

    #[test]
    fn test_template_variable_get_effective_value() {
        let var = TemplateVariable::with_default("name", VariableType::String, "default");

        assert_eq!(
            var.get_effective_value(Some(&"provided".to_string())),
            Some("provided".to_string())
        );
        assert_eq!(
            var.get_effective_value(None),
            Some("default".to_string())
        );

        let required_var = TemplateVariable::required("name", VariableType::String);
        assert_eq!(required_var.get_effective_value(None), None);
    }

    // =========================================================================
    // PromptTemplate Tests
    // =========================================================================

    #[test]
    fn test_prompt_template_new() {
        let template = PromptTemplate::new("greeting", "Hello, {{name}}!");
        assert_eq!(template.id, "greeting");
        assert_eq!(template.name, "greeting");
        assert_eq!(template.content, "Hello, {{name}}!");
        assert_eq!(template.version, "1.0.0");
    }

    #[test]
    fn test_prompt_template_builder_methods() {
        let template = PromptTemplate::new("test", "content")
            .with_name("Test Template")
            .with_version("2.0.0")
            .with_category("greeting")
            .with_tag("hello")
            .with_tag("world")
            .with_description("A test template")
            .with_author("Tester");

        assert_eq!(template.name, "Test Template");
        assert_eq!(template.version, "2.0.0");
        assert_eq!(template.category, Some("greeting".to_string()));
        assert_eq!(template.tags, vec!["hello", "world"]);
        assert_eq!(template.description, Some("A test template".to_string()));
        assert_eq!(template.author, Some("Tester".to_string()));
    }

    #[test]
    fn test_prompt_template_add_variable() {
        let mut template = PromptTemplate::new("test", "Hello, {{name}}!");
        template.add_variable(TemplateVariable::new("name"));
        assert!(template.variables.contains_key("name"));
    }

    #[test]
    fn test_prompt_template_with_variable() {
        let template = PromptTemplate::new("test", "Hello, {{name}}!")
            .with_variable(TemplateVariable::new("name"));
        assert!(template.variables.contains_key("name"));
    }

    #[test]
    fn test_prompt_template_extract_variable_names() {
        let template = PromptTemplate::new(
            "test",
            "Hello, {{name}}! Your order {{order_id}} is {{status}}. {{name}} will be notified.",
        );
        let names = template.extract_variable_names();
        assert_eq!(names, vec!["name", "order_id", "status"]);
    }

    #[test]
    fn test_prompt_template_undefined_variables() {
        let template = PromptTemplate::new("test", "Hello, {{name}}! {{undefined}}")
            .with_variable(TemplateVariable::new("name"));
        let undefined = template.undefined_variables();
        assert_eq!(undefined, vec!["undefined"]);
    }

    #[test]
    fn test_prompt_template_required_variables() {
        let template = PromptTemplate::new("test", "{{a}} {{b}}")
            .with_variable(TemplateVariable::required("a", VariableType::String))
            .with_variable(TemplateVariable::optional("b", VariableType::String));
        let required = template.required_variables();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].name, "a");
    }

    #[test]
    fn test_prompt_template_optional_variables() {
        let template = PromptTemplate::new("test", "{{a}} {{b}}")
            .with_variable(TemplateVariable::required("a", VariableType::String))
            .with_variable(TemplateVariable::optional("b", VariableType::String));
        let optional = template.optional_variables();
        assert_eq!(optional.len(), 1);
        assert_eq!(optional[0].name, "b");
    }

    #[test]
    fn test_prompt_template_validate_success() {
        let template = PromptTemplate::new("test", "Hello, {{name}}!")
            .with_variable(TemplateVariable::new("name"));
        assert!(template.validate().is_ok());
    }

    #[test]
    fn test_prompt_template_validate_undefined() {
        let template = PromptTemplate::new("test", "Hello, {{name}}!");
        let result = template.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].contains("Undefined variables"));
    }

    #[test]
    fn test_prompt_template_validate_empty() {
        let template = PromptTemplate::new("test", "   ");
        let result = template.validate();
        assert!(result.is_err());
    }

    // =========================================================================
    // InstantiatedPrompt Tests
    // =========================================================================

    #[test]
    fn test_instantiated_prompt_has_warnings() {
        let prompt = InstantiatedPrompt {
            content: "test".to_string(),
            template_id: "test".to_string(),
            template_version: "1.0.0".to_string(),
            variables_used: HashMap::new(),
            warnings: vec!["warning".to_string()],
            processing_time_ms: 10,
            defaults_applied: Vec::new(),
            unused_variables: Vec::new(),
        };
        assert!(prompt.has_warnings());
    }

    #[test]
    fn test_instantiated_prompt_used_defaults() {
        let prompt = InstantiatedPrompt {
            content: "test".to_string(),
            template_id: "test".to_string(),
            template_version: "1.0.0".to_string(),
            variables_used: HashMap::new(),
            warnings: Vec::new(),
            processing_time_ms: 10,
            defaults_applied: vec!["name".to_string()],
            unused_variables: Vec::new(),
        };
        assert!(prompt.used_defaults());
    }

    #[test]
    fn test_instantiated_prompt_has_unused() {
        let prompt = InstantiatedPrompt {
            content: "test".to_string(),
            template_id: "test".to_string(),
            template_version: "1.0.0".to_string(),
            variables_used: HashMap::new(),
            warnings: Vec::new(),
            processing_time_ms: 10,
            defaults_applied: Vec::new(),
            unused_variables: vec!["extra".to_string()],
        };
        assert!(prompt.has_unused_variables());
    }

    #[test]
    fn test_instantiated_prompt_variable_count() {
        let mut vars = HashMap::new();
        vars.insert("a".to_string(), "1".to_string());
        vars.insert("b".to_string(), "2".to_string());

        let prompt = InstantiatedPrompt {
            content: "test".to_string(),
            template_id: "test".to_string(),
            template_version: "1.0.0".to_string(),
            variables_used: vars,
            warnings: Vec::new(),
            processing_time_ms: 10,
            defaults_applied: Vec::new(),
            unused_variables: Vec::new(),
        };
        assert_eq!(prompt.variable_count(), 2);
    }

    // =========================================================================
    // TemplateRegistry Tests
    // =========================================================================

    #[test]
    fn test_registry_new() {
        let registry = TemplateRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = TemplateRegistry::new();
        let template = PromptTemplate::new("test", "content");
        registry.register(template);

        assert_eq!(registry.len(), 1);
        assert!(registry.contains("test"));
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_registry_template_ids() {
        let mut registry = TemplateRegistry::new();
        registry.register(PromptTemplate::new("a", ""));
        registry.register(PromptTemplate::new("b", ""));

        let ids = registry.template_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_registry_by_category() {
        let mut registry = TemplateRegistry::new();
        registry.register(PromptTemplate::new("a", "").with_category("cat1"));
        registry.register(PromptTemplate::new("b", "").with_category("cat1"));
        registry.register(PromptTemplate::new("c", "").with_category("cat2"));

        let cat1 = registry.by_category("cat1");
        assert_eq!(cat1.len(), 2);

        let cat2 = registry.by_category("cat2");
        assert_eq!(cat2.len(), 1);

        let cat3 = registry.by_category("nonexistent");
        assert_eq!(cat3.len(), 0);
    }

    #[test]
    fn test_registry_by_tag() {
        let mut registry = TemplateRegistry::new();
        registry.register(PromptTemplate::new("a", "").with_tag("tag1").with_tag("tag2"));
        registry.register(PromptTemplate::new("b", "").with_tag("tag1"));

        let tag1 = registry.by_tag("tag1");
        assert_eq!(tag1.len(), 2);

        let tag2 = registry.by_tag("tag2");
        assert_eq!(tag2.len(), 1);
    }

    #[test]
    fn test_registry_categories_and_tags() {
        let mut registry = TemplateRegistry::new();
        registry.register(
            PromptTemplate::new("a", "")
                .with_category("cat1")
                .with_tag("tag1")
                .with_tag("tag2"),
        );

        let categories = registry.categories();
        assert!(categories.contains(&"cat1"));

        let tags = registry.tags();
        assert!(tags.contains(&"tag1"));
        assert!(tags.contains(&"tag2"));
    }

    #[test]
    fn test_registry_remove() {
        let mut registry = TemplateRegistry::new();
        registry.register(PromptTemplate::new("test", "").with_category("cat"));

        let removed = registry.remove("test");
        assert!(removed.is_some());
        assert!(!registry.contains("test"));
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_search() {
        let mut registry = TemplateRegistry::new();
        registry.register(PromptTemplate::new("greeting", "").with_description("Say hello"));
        registry.register(PromptTemplate::new("farewell", "").with_description("Say goodbye"));

        let results = registry.search("hello");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "greeting");

        let results = registry.search("greeting");
        assert_eq!(results.len(), 1);

        let results = registry.search("say");
        assert_eq!(results.len(), 2);
    }

    // =========================================================================
    // TemplaterConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = TemplaterConfig::default();
        assert!(!config.allow_undefined);
        assert!(config.strict_types);
        assert!(config.trim_values);
        assert!(config.track_unused);
        assert_eq!(config.max_recursion_depth, 10);
    }

    #[test]
    fn test_config_strict() {
        let config = TemplaterConfig::strict();
        assert!(!config.allow_undefined);
        assert!(config.strict_types);
    }

    #[test]
    fn test_config_lenient() {
        let config = TemplaterConfig::lenient();
        assert!(config.allow_undefined);
        assert!(!config.strict_types);
    }

    // =========================================================================
    // TemplaterAgent Tests
    // =========================================================================

    #[test]
    fn test_agent_default() {
        let agent = TemplaterAgent::default();
        assert!(agent.registry().is_empty());
    }

    #[test]
    fn test_agent_new() {
        let agent = TemplaterAgent::new(TemplaterConfig::default());
        assert!(agent.registry().is_empty());
    }

    #[test]
    fn test_agent_builder() {
        let template = PromptTemplate::new("test", "Hello!");
        let agent = TemplaterAgent::builder()
            .config(TemplaterConfig::lenient())
            .add_template(template)
            .build();

        assert_eq!(agent.registry().len(), 1);
        assert!(agent.config().allow_undefined);
    }

    #[test]
    fn test_agent_builder_strict() {
        let agent = TemplaterAgent::builder().strict().build();
        assert!(!agent.config().allow_undefined);
        assert!(agent.config().strict_types);
    }

    #[test]
    fn test_agent_builder_lenient() {
        let agent = TemplaterAgent::builder().lenient().build();
        assert!(agent.config().allow_undefined);
        assert!(!agent.config().strict_types);
    }

    #[test]
    fn test_agent_register_template() {
        let mut agent = TemplaterAgent::default();
        agent.register_template(PromptTemplate::new("test", "content"));
        assert!(agent.get_template("test").is_some());
    }

    #[tokio::test]
    async fn test_agent_instantiate_basic() {
        let template = PromptTemplate::new("greeting", "Hello, {{name}}!")
            .with_variable(TemplateVariable::new("name"));

        let agent = TemplaterAgent::builder().add_template(template).build();

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "World".to_string());

        let result = agent.instantiate("greeting", &vars).await.unwrap();
        assert_eq!(result.content, "Hello, World!");
        assert_eq!(result.template_id, "greeting");
    }

    #[tokio::test]
    async fn test_agent_instantiate_with_default() {
        let template = PromptTemplate::new("greeting", "Hello, {{name}}!")
            .with_variable(TemplateVariable::with_default(
                "name",
                VariableType::String,
                "Guest",
            ));

        let agent = TemplaterAgent::builder().add_template(template).build();

        let vars = HashMap::new();
        let result = agent.instantiate("greeting", &vars).await.unwrap();

        assert_eq!(result.content, "Hello, Guest!");
        assert!(result.defaults_applied.contains(&"name".to_string()));
    }

    #[tokio::test]
    async fn test_agent_instantiate_missing_required() {
        let template = PromptTemplate::new("greeting", "Hello, {{name}}!")
            .with_variable(TemplateVariable::required("name", VariableType::String));

        let agent = TemplaterAgent::builder().add_template(template).build();

        let vars = HashMap::new();
        let result = agent.instantiate("greeting", &vars).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_instantiate_type_validation() {
        let template = PromptTemplate::new("count", "Count: {{n}}")
            .with_variable(TemplateVariable::required("n", VariableType::Number));

        let agent = TemplaterAgent::builder().strict().add_template(template).build();

        let mut vars = HashMap::new();
        vars.insert("n".to_string(), "not a number".to_string());

        let result = agent.instantiate("count", &vars).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_instantiate_type_warning_lenient() {
        let template = PromptTemplate::new("count", "Count: {{n}}")
            .with_variable(TemplateVariable::required("n", VariableType::Number));

        let agent = TemplaterAgent::builder().lenient().add_template(template).build();

        let mut vars = HashMap::new();
        vars.insert("n".to_string(), "not a number".to_string());

        let result = agent.instantiate("count", &vars).await.unwrap();
        assert!(result.has_warnings());
    }

    #[tokio::test]
    async fn test_agent_instantiate_multiple_variables() {
        let template = PromptTemplate::new("full", "{{greeting}}, {{name}}! Today is {{day}}.")
            .with_variable(TemplateVariable::required("greeting", VariableType::String))
            .with_variable(TemplateVariable::required("name", VariableType::String))
            .with_variable(TemplateVariable::with_default(
                "day",
                VariableType::String,
                "Monday",
            ));

        let agent = TemplaterAgent::builder().add_template(template).build();

        let mut vars = HashMap::new();
        vars.insert("greeting".to_string(), "Hello".to_string());
        vars.insert("name".to_string(), "Alice".to_string());

        let result = agent.instantiate("full", &vars).await.unwrap();
        assert_eq!(result.content, "Hello, Alice! Today is Monday.");
    }

    #[tokio::test]
    async fn test_agent_instantiate_unused_variables() {
        let template = PromptTemplate::new("simple", "Hello!")
            .with_variable(TemplateVariable::optional("unused", VariableType::String));

        let agent = TemplaterAgent::builder().add_template(template).build();

        let mut vars = HashMap::new();
        vars.insert("extra".to_string(), "value".to_string());

        let result = agent.instantiate("simple", &vars).await.unwrap();
        assert!(result.has_unused_variables());
        assert!(result.unused_variables.contains(&"extra".to_string()));
    }

    #[tokio::test]
    async fn test_agent_instantiate_template_not_found() {
        let agent = TemplaterAgent::default();

        let vars = HashMap::new();
        let result = agent.instantiate("nonexistent", &vars).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_instantiate_content() {
        let agent = TemplaterAgent::default();

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "World".to_string());

        let result = agent
            .instantiate_content("Hello, {{name}}!", &vars)
            .await
            .unwrap();
        assert_eq!(result.content, "Hello, World!");
    }

    #[test]
    fn test_agent_preview() {
        let template = PromptTemplate::new("test", "Hello, {{name}}!");
        let agent = TemplaterAgent::builder().add_template(template).build();

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Preview".to_string());

        let preview = agent.preview("test", &vars).unwrap();
        assert_eq!(preview, "Hello, Preview!");
    }

    #[test]
    fn test_agent_validate_template() {
        let template = PromptTemplate::new("valid", "Hello, {{name}}!")
            .with_variable(TemplateVariable::new("name"));
        let agent = TemplaterAgent::builder().add_template(template).build();

        assert!(agent.validate_template("valid").is_ok());
    }

    #[test]
    fn test_agent_list_templates() {
        let agent = TemplaterAgent::builder()
            .add_template(PromptTemplate::new("a", ""))
            .add_template(PromptTemplate::new("b", ""))
            .build();

        let list = agent.list_templates();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_agent_search_templates() {
        let agent = TemplaterAgent::builder()
            .add_template(PromptTemplate::new("greeting", "").with_description("Say hello"))
            .add_template(PromptTemplate::new("farewell", "").with_description("Say goodbye"))
            .build();

        let results = agent.search_templates("hello");
        assert_eq!(results.len(), 1);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_variable_type_serialization() {
        let vt = VariableType::enumeration(vec!["a", "b"]);
        let json = serde_json::to_string(&vt).unwrap();
        let deserialized: VariableType = serde_json::from_str(&json).unwrap();

        if let VariableType::Enum { allowed_values } = deserialized {
            assert_eq!(allowed_values, vec!["a", "b"]);
        } else {
            panic!("Expected Enum");
        }
    }

    #[test]
    fn test_template_variable_serialization() {
        let var = TemplateVariable::with_default("name", VariableType::String, "default")
            .with_description("A name")
            .with_validation(r"^\w+$");

        let json = serde_json::to_string(&var).unwrap();
        let deserialized: TemplateVariable = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "name");
        assert_eq!(deserialized.default, Some("default".to_string()));
    }

    #[test]
    fn test_prompt_template_serialization() {
        let template = PromptTemplate::new("test", "Hello, {{name}}!")
            .with_category("greeting")
            .with_tag("hello")
            .with_variable(TemplateVariable::new("name"));

        let json = serde_json::to_string(&template).unwrap();
        let deserialized: PromptTemplate = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "test");
        assert!(deserialized.variables.contains_key("name"));
    }

    #[test]
    fn test_config_serialization() {
        let config = TemplaterConfig::lenient();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TemplaterConfig = serde_json::from_str(&json).unwrap();

        assert!(deserialized.allow_undefined);
        assert!(!deserialized.strict_types);
    }
}
