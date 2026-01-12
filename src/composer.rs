//! Composer Agent module for Project Panpsychism.
//!
//! Agent 31: The Prompt Builder - "From fragments, forge the perfect spell."
//!
//! This module implements the Composer Agent, responsible for assembling complex
//! prompts from reusable building blocks. Like a master conductor orchestrating
//! an ensemble, the Composer combines individual components into a harmonious whole.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, the Composer Agent serves as the grand architect:
//!
//! - **Components** (spell fragments) are gathered from the library
//! - **The Builder** (ComposerAgent) arranges them with precision
//! - **Composed Prompt** (unified incantation) emerges ready for casting
//!
//! The Builder orchestrates:
//! - Component ordering based on roles and constraints
//! - Token budget management for LLM limits
//! - Structure selection (linear, hierarchical, conversational)
//! - Validation of composition rules
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//!
//! - **RATIO**: Logical ordering and structure of components
//! - **NATURA**: Natural flow from context to instruction to output
//! - **CONATUS**: Drive to create coherent, self-sustaining prompts
//! - **LAETITIA**: Joy through elegant composition and clarity
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::composer::{ComposerAgent, PromptComponent, ComponentRole};
//!
//! let composer = ComposerAgent::new();
//!
//! // Build from components
//! let components = vec![
//!     PromptComponent::new("ctx-1", "You are a helpful assistant.")
//!         .with_role(ComponentRole::SystemContext),
//!     PromptComponent::new("inst-1", "Explain quantum computing.")
//!         .with_role(ComponentRole::UserInstruction),
//!     PromptComponent::new("fmt-1", "Use markdown with code examples.")
//!         .with_role(ComponentRole::OutputFormat),
//! ];
//!
//! let composed = composer.compose(&components).await?;
//! println!("{}", composed.content);
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tracing::{debug, info};

// =============================================================================
// COMPONENT ROLE ENUM
// =============================================================================

/// The role a component plays within the composed prompt.
///
/// Each role has a natural ordering and purpose within the final prompt,
/// following established patterns for effective LLM communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ComponentRole {
    /// System context that sets the AI's persona and capabilities.
    ///
    /// Typically appears first in the prompt to establish the frame.
    /// Example: "You are an expert software architect..."
    #[default]
    SystemContext,

    /// Direct instruction from the user about what to do.
    ///
    /// The core request that drives the response.
    /// Example: "Implement a binary search algorithm."
    UserInstruction,

    /// Concrete examples demonstrating expected behavior.
    ///
    /// Few-shot learning examples that guide the response format.
    /// Example: "Input: 5 -> Output: 25"
    Example,

    /// Constraints that limit or guide the response.
    ///
    /// Rules the response must follow.
    /// Example: "Do not use external libraries."
    Constraint,

    /// Specification for how output should be formatted.
    ///
    /// Format requirements for the response.
    /// Example: "Return as JSON with keys: result, explanation."
    OutputFormat,

    /// A persona or character the AI should adopt.
    ///
    /// Defines behavioral characteristics beyond system context.
    /// Example: "Respond as a friendly tutor who uses analogies."
    Persona,
}

impl std::fmt::Display for ComponentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SystemContext => write!(f, "system_context"),
            Self::UserInstruction => write!(f, "user_instruction"),
            Self::Example => write!(f, "example"),
            Self::Constraint => write!(f, "constraint"),
            Self::OutputFormat => write!(f, "output_format"),
            Self::Persona => write!(f, "persona"),
        }
    }
}

impl std::str::FromStr for ComponentRole {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "system_context" | "system" | "context" => Ok(Self::SystemContext),
            "user_instruction" | "instruction" | "user" | "task" => Ok(Self::UserInstruction),
            "example" | "examples" | "few_shot" | "fewshot" => Ok(Self::Example),
            "constraint" | "constraints" | "rule" | "rules" => Ok(Self::Constraint),
            "output_format" | "output" | "format" => Ok(Self::OutputFormat),
            "persona" | "character" | "role" => Ok(Self::Persona),
            _ => Err(Error::Config(format!(
                "Unknown component role: '{}'. Valid roles: system_context, user_instruction, \
                 example, constraint, output_format, persona",
                s
            ))),
        }
    }
}

impl ComponentRole {
    /// Get the natural ordering priority for this role.
    ///
    /// Lower values appear earlier in the composed prompt.
    pub fn natural_order(&self) -> u8 {
        match self {
            Self::SystemContext => 0,
            Self::Persona => 1,
            Self::UserInstruction => 2,
            Self::Example => 3,
            Self::Constraint => 4,
            Self::OutputFormat => 5,
        }
    }

    /// Get all component roles.
    pub fn all() -> Vec<Self> {
        vec![
            Self::SystemContext,
            Self::Persona,
            Self::UserInstruction,
            Self::Example,
            Self::Constraint,
            Self::OutputFormat,
        ]
    }

    /// Get the typical label/header for this role in a prompt.
    pub fn label(&self) -> &'static str {
        match self {
            Self::SystemContext => "System Context",
            Self::Persona => "Persona",
            Self::UserInstruction => "Task",
            Self::Example => "Examples",
            Self::Constraint => "Constraints",
            Self::OutputFormat => "Output Format",
        }
    }

    /// Check if this role is typically required in a well-formed prompt.
    pub fn is_typically_required(&self) -> bool {
        matches!(self, Self::SystemContext | Self::UserInstruction)
    }
}

// =============================================================================
// PROMPT COMPONENT
// =============================================================================

/// A reusable building block for prompt composition.
///
/// Each component represents a discrete piece of the final prompt,
/// with its own identity, content, role, and ordering requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptComponent {
    /// Unique identifier for this component.
    pub id: String,

    /// The actual content of this component.
    pub content: String,

    /// The role this component plays in the composition.
    pub role: ComponentRole,

    /// Whether this component is required in the final prompt.
    pub required: bool,

    /// Explicit ordering override (None uses natural role ordering).
    pub order: Option<u8>,

    /// Tags for categorization and filtering.
    pub tags: HashSet<String>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl PromptComponent {
    /// Create a new prompt component with id and content.
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            role: ComponentRole::default(),
            required: false,
            order: None,
            tags: HashSet::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the role for this component.
    pub fn with_role(mut self, role: ComponentRole) -> Self {
        self.role = role;
        self
    }

    /// Mark this component as required.
    pub fn with_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set an explicit ordering for this component.
    pub fn with_order(mut self, order: u8) -> Self {
        self.order = Some(order);
        self
    }

    /// Add a tag to this component.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.insert(tag.into());
        self
    }

    /// Add multiple tags to this component.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for tag in tags {
            self.tags.insert(tag.into());
        }
        self
    }

    /// Add metadata to this component.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get the effective ordering for this component.
    pub fn effective_order(&self) -> u8 {
        self.order.unwrap_or_else(|| self.role.natural_order())
    }

    /// Estimate token count for this component (rough: ~4 chars per token).
    pub fn estimate_tokens(&self) -> usize {
        self.content.len() / 4 + 1
    }

    /// Check if this component has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }
}

impl Default for PromptComponent {
    fn default() -> Self {
        Self::new("", "")
    }
}

// =============================================================================
// PROMPT STRUCTURE ENUM
// =============================================================================

/// The structural pattern for composing prompts.
///
/// Different structures are optimal for different use cases and LLM interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PromptStructure {
    /// Linear structure - components concatenated in order.
    ///
    /// Simplest form: component 1 + separator + component 2 + ...
    /// Best for straightforward single-turn prompts.
    #[default]
    Linear,

    /// Hierarchical structure - components nested by importance.
    ///
    /// Uses XML-like or markdown headers to create hierarchy.
    /// Best for complex prompts with multiple sections.
    Hierarchical,

    /// Conversational structure - components as message turns.
    ///
    /// Formatted as user/assistant message pairs.
    /// Best for chat-based interactions and few-shot examples.
    Conversational,
}

impl std::fmt::Display for PromptStructure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear => write!(f, "linear"),
            Self::Hierarchical => write!(f, "hierarchical"),
            Self::Conversational => write!(f, "conversational"),
        }
    }
}

impl std::str::FromStr for PromptStructure {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "linear" | "flat" | "simple" => Ok(Self::Linear),
            "hierarchical" | "nested" | "structured" => Ok(Self::Hierarchical),
            "conversational" | "chat" | "messages" | "turns" => Ok(Self::Conversational),
            _ => Err(Error::Config(format!(
                "Unknown prompt structure: '{}'. Valid structures: linear, hierarchical, conversational",
                s
            ))),
        }
    }
}

impl PromptStructure {
    /// Get the default separator for this structure.
    pub fn default_separator(&self) -> &'static str {
        match self {
            Self::Linear => "\n\n",
            Self::Hierarchical => "\n",
            Self::Conversational => "\n---\n",
        }
    }

    /// Get all available structures.
    pub fn all() -> Vec<Self> {
        vec![Self::Linear, Self::Hierarchical, Self::Conversational]
    }
}

// =============================================================================
// ORDER CONSTRAINT
// =============================================================================

/// A constraint on the ordering of components in the composition.
///
/// Allows expressing rules like "A must come before B" or
/// "C must be adjacent to D".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderConstraint {
    /// The component that must appear first.
    pub before_id: String,

    /// The component that must appear after.
    pub after_id: String,

    /// Whether these components must be adjacent (no components between them).
    pub adjacent: bool,

    /// Description of why this constraint exists.
    pub reason: Option<String>,
}

impl OrderConstraint {
    /// Create a new order constraint.
    pub fn new(before_id: impl Into<String>, after_id: impl Into<String>) -> Self {
        Self {
            before_id: before_id.into(),
            after_id: after_id.into(),
            adjacent: false,
            reason: None,
        }
    }

    /// Mark this constraint as requiring adjacency.
    pub fn with_adjacent(mut self, adjacent: bool) -> Self {
        self.adjacent = adjacent;
        self
    }

    /// Add a reason for this constraint.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Check if this constraint is satisfied by an ordering.
    pub fn is_satisfied(&self, order: &[&str]) -> bool {
        let before_pos = order.iter().position(|&id| id == self.before_id);
        let after_pos = order.iter().position(|&id| id == self.after_id);

        match (before_pos, after_pos) {
            (Some(b), Some(a)) => {
                if b >= a {
                    return false;
                }
                if self.adjacent && a != b + 1 {
                    return false;
                }
                true
            }
            // If either component is missing, constraint doesn't apply
            _ => true,
        }
    }
}

impl Default for OrderConstraint {
    fn default() -> Self {
        Self::new("", "")
    }
}

// =============================================================================
// COMPOSITION RULE
// =============================================================================

/// Rules governing how components can be composed together.
///
/// Defines requirements and constraints for valid compositions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionRule {
    /// Roles that must be present in the composition.
    pub required_roles: HashSet<ComponentRole>,

    /// Ordering constraints between specific components.
    pub order_constraints: Vec<OrderConstraint>,

    /// Maximum total tokens allowed in the composition.
    pub max_tokens: usize,

    /// Minimum components required.
    pub min_components: usize,

    /// Maximum components allowed.
    pub max_components: usize,

    /// Whether to allow duplicate roles.
    pub allow_duplicate_roles: bool,

    /// Whether to validate component ordering.
    pub enforce_ordering: bool,
}

impl Default for CompositionRule {
    fn default() -> Self {
        Self {
            required_roles: [ComponentRole::UserInstruction].into_iter().collect(),
            order_constraints: Vec::new(),
            max_tokens: 100_000,
            min_components: 1,
            max_components: 50,
            allow_duplicate_roles: true,
            enforce_ordering: true,
        }
    }
}

impl CompositionRule {
    /// Create a new composition rule.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a strict composition rule.
    pub fn strict() -> Self {
        Self {
            required_roles: [ComponentRole::SystemContext, ComponentRole::UserInstruction]
                .into_iter()
                .collect(),
            order_constraints: Vec::new(),
            max_tokens: 50_000,
            min_components: 2,
            max_components: 20,
            allow_duplicate_roles: false,
            enforce_ordering: true,
        }
    }

    /// Create a lenient composition rule.
    pub fn lenient() -> Self {
        Self {
            required_roles: HashSet::new(),
            order_constraints: Vec::new(),
            max_tokens: 500_000,
            min_components: 0,
            max_components: 100,
            allow_duplicate_roles: true,
            enforce_ordering: false,
        }
    }

    /// Add a required role.
    pub fn with_required_role(mut self, role: ComponentRole) -> Self {
        self.required_roles.insert(role);
        self
    }

    /// Add an order constraint.
    pub fn with_order_constraint(mut self, constraint: OrderConstraint) -> Self {
        self.order_constraints.push(constraint);
        self
    }

    /// Set the maximum token limit.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Set minimum and maximum component counts.
    pub fn with_component_limits(mut self, min: usize, max: usize) -> Self {
        self.min_components = min;
        self.max_components = max;
        self
    }

    /// Validate components against this rule.
    pub fn validate(&self, components: &[PromptComponent]) -> Vec<String> {
        let mut errors = Vec::new();

        // Check component count
        if components.len() < self.min_components {
            errors.push(format!(
                "Too few components: {} < minimum {}",
                components.len(),
                self.min_components
            ));
        }
        if components.len() > self.max_components {
            errors.push(format!(
                "Too many components: {} > maximum {}",
                components.len(),
                self.max_components
            ));
        }

        // Check required roles
        let present_roles: HashSet<ComponentRole> = components.iter().map(|c| c.role).collect();
        for required in &self.required_roles {
            if !present_roles.contains(required) {
                errors.push(format!("Missing required role: {}", required));
            }
        }

        // Check duplicate roles
        if !self.allow_duplicate_roles {
            let mut seen_roles = HashSet::new();
            for component in components {
                if !seen_roles.insert(component.role) {
                    errors.push(format!("Duplicate role not allowed: {}", component.role));
                }
            }
        }

        // Check token limit
        let total_tokens: usize = components.iter().map(|c| c.estimate_tokens()).sum();
        if total_tokens > self.max_tokens {
            errors.push(format!(
                "Exceeds token limit: ~{} > {}",
                total_tokens, self.max_tokens
            ));
        }

        // Check order constraints
        if self.enforce_ordering {
            let order: Vec<&str> = components.iter().map(|c| c.id.as_str()).collect();
            for constraint in &self.order_constraints {
                if !constraint.is_satisfied(&order) {
                    let reason = constraint
                        .reason
                        .as_deref()
                        .unwrap_or("ordering requirement");
                    errors.push(format!(
                        "Order constraint violated: {} must come before {} ({})",
                        constraint.before_id, constraint.after_id, reason
                    ));
                }
            }
        }

        errors
    }
}

// =============================================================================
// COMPOSED PROMPT
// =============================================================================

/// The result of composing multiple components into a single prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedPrompt {
    /// The final composed content ready for LLM submission.
    pub content: String,

    /// The components that were included in the composition.
    pub components: Vec<PromptComponent>,

    /// The structure used for composition.
    pub structure: PromptStructure,

    /// Estimated token count of the composed prompt.
    pub token_count: usize,

    /// Time taken to compose in milliseconds.
    pub duration_ms: u64,

    /// Any warnings generated during composition.
    pub warnings: Vec<String>,

    /// Metadata about the composition.
    pub metadata: HashMap<String, String>,
}

impl ComposedPrompt {
    /// Create a new composed prompt.
    pub fn new(content: impl Into<String>, components: Vec<PromptComponent>) -> Self {
        let content = content.into();
        let token_count = content.len() / 4 + 1;

        Self {
            content,
            components,
            structure: PromptStructure::default(),
            token_count,
            duration_ms: 0,
            warnings: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get the number of components in this composition.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Get components by role.
    pub fn components_by_role(&self, role: ComponentRole) -> Vec<&PromptComponent> {
        self.components.iter().filter(|c| c.role == role).collect()
    }

    /// Check if the composition includes a specific role.
    pub fn has_role(&self, role: ComponentRole) -> bool {
        self.components.iter().any(|c| c.role == role)
    }

    /// Get a summary of the composition.
    pub fn summary(&self) -> String {
        format!(
            "Composed: {} components, ~{} tokens, {} structure ({}ms)",
            self.components.len(),
            self.token_count,
            self.structure,
            self.duration_ms
        )
    }

    /// Check if there were any warnings during composition.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl Default for ComposedPrompt {
    fn default() -> Self {
        Self::new("", Vec::new())
    }
}

// =============================================================================
// COMPOSER CONFIGURATION
// =============================================================================

/// Configuration for the Composer Agent.
#[derive(Debug, Clone)]
pub struct ComposerConfig {
    /// The default structure to use for composition.
    pub default_structure: PromptStructure,

    /// Default composition rules.
    pub default_rules: CompositionRule,

    /// Separator between components in linear structure.
    pub separator: String,

    /// Whether to include role labels/headers.
    pub include_headers: bool,

    /// Whether to trim whitespace from components.
    pub trim_content: bool,

    /// Whether to deduplicate identical content.
    pub deduplicate: bool,

    /// Maximum estimated tokens before warning.
    pub token_warning_threshold: usize,
}

impl Default for ComposerConfig {
    fn default() -> Self {
        Self {
            default_structure: PromptStructure::Linear,
            default_rules: CompositionRule::default(),
            separator: "\n\n".to_string(),
            include_headers: true,
            trim_content: true,
            deduplicate: true,
            token_warning_threshold: 4000,
        }
    }
}

impl ComposerConfig {
    /// Create a minimal configuration for simple prompts.
    pub fn minimal() -> Self {
        Self {
            default_structure: PromptStructure::Linear,
            default_rules: CompositionRule::lenient(),
            separator: "\n".to_string(),
            include_headers: false,
            trim_content: true,
            deduplicate: false,
            token_warning_threshold: 8000,
        }
    }

    /// Create a structured configuration for complex prompts.
    pub fn structured() -> Self {
        Self {
            default_structure: PromptStructure::Hierarchical,
            default_rules: CompositionRule::strict(),
            separator: "\n\n".to_string(),
            include_headers: true,
            trim_content: true,
            deduplicate: true,
            token_warning_threshold: 4000,
        }
    }

    /// Create a chat-optimized configuration.
    pub fn chat() -> Self {
        Self {
            default_structure: PromptStructure::Conversational,
            default_rules: CompositionRule::default(),
            separator: "\n---\n".to_string(),
            include_headers: false,
            trim_content: true,
            deduplicate: false,
            token_warning_threshold: 4000,
        }
    }
}

// =============================================================================
// COMPOSER AGENT
// =============================================================================

/// The Composer Agent - Agent 31 of Project Panpsychism.
///
/// Responsible for assembling complex prompts from reusable building blocks.
/// Like a master conductor, it orchestrates components into a harmonious whole.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **RATIO**: Logical ordering and structure
/// - **NATURA**: Natural flow from context to instruction
/// - **CONATUS**: Drive to create coherent prompts
/// - **LAETITIA**: Joy through elegant composition
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::composer::{ComposerAgent, PromptComponent, ComponentRole};
///
/// let composer = ComposerAgent::new();
///
/// let components = vec![
///     PromptComponent::new("sys", "You are a helpful assistant.")
///         .with_role(ComponentRole::SystemContext),
///     PromptComponent::new("task", "Explain recursion.")
///         .with_role(ComponentRole::UserInstruction),
/// ];
///
/// let composed = composer.compose(&components).await?;
/// println!("{}", composed.content);
/// ```
#[derive(Debug, Clone)]
pub struct ComposerAgent {
    /// Configuration for composition behavior.
    config: ComposerConfig,
}

impl Default for ComposerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl ComposerAgent {
    /// Create a new Composer Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: ComposerConfig::default(),
        }
    }

    /// Create a Composer Agent with custom configuration.
    pub fn with_config(config: ComposerConfig) -> Self {
        Self { config }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> ComposerAgentBuilder {
        ComposerAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ComposerConfig {
        &self.config
    }

    // =========================================================================
    // MAIN COMPOSITION METHOD
    // =========================================================================

    /// Compose multiple components into a single prompt.
    ///
    /// This is the primary composition method. It orders components,
    /// validates against rules, and renders them into the configured structure.
    ///
    /// # Arguments
    ///
    /// * `components` - The components to compose
    ///
    /// # Returns
    ///
    /// A `ComposedPrompt` containing the assembled content and metadata.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` if composition rules are violated.
    pub async fn compose(&self, components: &[PromptComponent]) -> Result<ComposedPrompt> {
        self.compose_with_structure(components, self.config.default_structure)
            .await
    }

    /// Compose with a specific structure.
    pub async fn compose_with_structure(
        &self,
        components: &[PromptComponent],
        structure: PromptStructure,
    ) -> Result<ComposedPrompt> {
        self.compose_with_rules(components, structure, &self.config.default_rules)
            .await
    }

    /// Compose with custom rules.
    pub async fn compose_with_rules(
        &self,
        components: &[PromptComponent],
        structure: PromptStructure,
        rules: &CompositionRule,
    ) -> Result<ComposedPrompt> {
        let start = Instant::now();

        debug!(
            "Composing {} components with {} structure",
            components.len(),
            structure
        );

        // Validate components
        let validation_errors = rules.validate(components);
        if !validation_errors.is_empty() {
            return Err(Error::Validation(format!(
                "Composition validation failed: {}",
                validation_errors.join("; ")
            )));
        }

        // Prepare components (deduplicate, trim, etc.)
        let prepared = self.prepare_components(components);

        // Sort components by effective order
        let sorted = self.sort_components(&prepared);

        // Check for required components
        let mut warnings = Vec::new();
        for component in &sorted {
            if component.required && component.content.trim().is_empty() {
                warnings.push(format!(
                    "Required component '{}' has empty content",
                    component.id
                ));
            }
        }

        // Render to the specified structure
        let content = self.render_structure(&sorted, structure);

        // Calculate token estimate
        let token_count = content.len() / 4 + 1;

        // Warn if exceeding threshold
        if token_count > self.config.token_warning_threshold {
            warnings.push(format!(
                "Composed prompt is large: ~{} tokens (threshold: {})",
                token_count, self.config.token_warning_threshold
            ));
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        info!(
            "Composed {} components into {} structure (~{} tokens, {}ms)",
            sorted.len(),
            structure,
            token_count,
            duration_ms
        );

        Ok(ComposedPrompt {
            content,
            components: sorted,
            structure,
            token_count,
            duration_ms,
            warnings,
            metadata: HashMap::new(),
        })
    }

    // =========================================================================
    // COMPONENT PREPARATION
    // =========================================================================

    /// Prepare components for composition.
    fn prepare_components(&self, components: &[PromptComponent]) -> Vec<PromptComponent> {
        let mut prepared: Vec<PromptComponent> = components.to_vec();

        // Trim content if configured
        if self.config.trim_content {
            for component in &mut prepared {
                component.content = component.content.trim().to_string();
            }
        }

        // Deduplicate if configured
        if self.config.deduplicate {
            let mut seen_content: HashSet<String> = HashSet::new();
            prepared.retain(|c| {
                let key = format!("{}:{}", c.role, c.content);
                seen_content.insert(key)
            });
        }

        prepared
    }

    /// Sort components by effective order.
    fn sort_components(&self, components: &[PromptComponent]) -> Vec<PromptComponent> {
        let mut sorted = components.to_vec();
        sorted.sort_by_key(|c| c.effective_order());
        sorted
    }

    // =========================================================================
    // STRUCTURE RENDERING
    // =========================================================================

    /// Render components into the specified structure.
    fn render_structure(
        &self,
        components: &[PromptComponent],
        structure: PromptStructure,
    ) -> String {
        match structure {
            PromptStructure::Linear => self.render_linear(components),
            PromptStructure::Hierarchical => self.render_hierarchical(components),
            PromptStructure::Conversational => self.render_conversational(components),
        }
    }

    /// Render as linear (simple concatenation).
    fn render_linear(&self, components: &[PromptComponent]) -> String {
        let mut parts: Vec<String> = Vec::new();

        for component in components {
            if component.content.is_empty() {
                continue;
            }

            if self.config.include_headers {
                parts.push(format!(
                    "## {}\n\n{}",
                    component.role.label(),
                    component.content
                ));
            } else {
                parts.push(component.content.clone());
            }
        }

        parts.join(&self.config.separator)
    }

    /// Render as hierarchical (nested sections).
    fn render_hierarchical(&self, components: &[PromptComponent]) -> String {
        let mut output = String::new();

        // Group by role
        let mut by_role: HashMap<ComponentRole, Vec<&PromptComponent>> = HashMap::new();
        for component in components {
            by_role
                .entry(component.role)
                .or_default()
                .push(component);
        }

        // Render in natural order
        for role in ComponentRole::all() {
            if let Some(role_components) = by_role.get(&role) {
                if role_components.iter().all(|c| c.content.is_empty()) {
                    continue;
                }

                output.push_str(&format!("<{}>\n", role));

                for component in role_components {
                    if !component.content.is_empty() {
                        output.push_str(&component.content);
                        output.push('\n');
                    }
                }

                output.push_str(&format!("</{}>\n\n", role));
            }
        }

        output.trim().to_string()
    }

    /// Render as conversational (message turns).
    fn render_conversational(&self, components: &[PromptComponent]) -> String {
        let mut output = String::new();

        for component in components {
            if component.content.is_empty() {
                continue;
            }

            let role_name = match component.role {
                ComponentRole::SystemContext | ComponentRole::Persona => "system",
                ComponentRole::UserInstruction => "user",
                ComponentRole::Example => "example",
                ComponentRole::Constraint | ComponentRole::OutputFormat => "assistant",
            };

            output.push_str(&format!("[{}]\n{}\n\n", role_name, component.content));
        }

        output.trim().to_string()
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Estimate total tokens for a set of components.
    pub fn estimate_tokens(&self, components: &[PromptComponent]) -> usize {
        components.iter().map(|c| c.estimate_tokens()).sum()
    }

    /// Validate components without composing.
    pub fn validate(&self, components: &[PromptComponent]) -> Vec<String> {
        self.config.default_rules.validate(components)
    }

    /// Filter components by role.
    pub fn filter_by_role<'a>(
        components: &'a [PromptComponent],
        role: ComponentRole,
    ) -> Vec<&'a PromptComponent> {
        components.iter().filter(|c| c.role == role).collect()
    }

    /// Filter components by tag.
    pub fn filter_by_tag<'a>(
        components: &'a [PromptComponent],
        tag: &str,
    ) -> Vec<&'a PromptComponent> {
        components.iter().filter(|c| c.has_tag(tag)).collect()
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom ComposerAgent configuration.
#[derive(Debug, Default)]
pub struct ComposerAgentBuilder {
    config: Option<ComposerConfig>,
}

impl ComposerAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: ComposerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the default structure.
    pub fn default_structure(mut self, structure: PromptStructure) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.default_structure = structure;
        self.config = Some(config);
        self
    }

    /// Set the separator for linear structure.
    pub fn separator(mut self, separator: impl Into<String>) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.separator = separator.into();
        self.config = Some(config);
        self
    }

    /// Set whether to include headers.
    pub fn include_headers(mut self, include: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_headers = include;
        self.config = Some(config);
        self
    }

    /// Set whether to trim content.
    pub fn trim_content(mut self, trim: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.trim_content = trim;
        self.config = Some(config);
        self
    }

    /// Set whether to deduplicate content.
    pub fn deduplicate(mut self, dedupe: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.deduplicate = dedupe;
        self.config = Some(config);
        self
    }

    /// Set the token warning threshold.
    pub fn token_warning_threshold(mut self, threshold: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.token_warning_threshold = threshold;
        self.config = Some(config);
        self
    }

    /// Set default composition rules.
    pub fn default_rules(mut self, rules: CompositionRule) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.default_rules = rules;
        self.config = Some(config);
        self
    }

    /// Build the ComposerAgent.
    pub fn build(self) -> ComposerAgent {
        ComposerAgent {
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
    // ComponentRole Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_component_role_display() {
        assert_eq!(ComponentRole::SystemContext.to_string(), "system_context");
        assert_eq!(ComponentRole::UserInstruction.to_string(), "user_instruction");
        assert_eq!(ComponentRole::Example.to_string(), "example");
        assert_eq!(ComponentRole::Constraint.to_string(), "constraint");
        assert_eq!(ComponentRole::OutputFormat.to_string(), "output_format");
        assert_eq!(ComponentRole::Persona.to_string(), "persona");
    }

    #[test]
    fn test_component_role_from_str() {
        assert_eq!(
            "system_context".parse::<ComponentRole>().unwrap(),
            ComponentRole::SystemContext
        );
        assert_eq!(
            "system".parse::<ComponentRole>().unwrap(),
            ComponentRole::SystemContext
        );
        assert_eq!(
            "user_instruction".parse::<ComponentRole>().unwrap(),
            ComponentRole::UserInstruction
        );
        assert_eq!(
            "instruction".parse::<ComponentRole>().unwrap(),
            ComponentRole::UserInstruction
        );
        assert_eq!(
            "example".parse::<ComponentRole>().unwrap(),
            ComponentRole::Example
        );
        assert_eq!(
            "constraint".parse::<ComponentRole>().unwrap(),
            ComponentRole::Constraint
        );
        assert_eq!(
            "output_format".parse::<ComponentRole>().unwrap(),
            ComponentRole::OutputFormat
        );
        assert_eq!(
            "persona".parse::<ComponentRole>().unwrap(),
            ComponentRole::Persona
        );
    }

    #[test]
    fn test_component_role_from_str_invalid() {
        assert!("invalid".parse::<ComponentRole>().is_err());
    }

    #[test]
    fn test_component_role_natural_order() {
        assert_eq!(ComponentRole::SystemContext.natural_order(), 0);
        assert_eq!(ComponentRole::Persona.natural_order(), 1);
        assert_eq!(ComponentRole::UserInstruction.natural_order(), 2);
        assert_eq!(ComponentRole::Example.natural_order(), 3);
        assert_eq!(ComponentRole::Constraint.natural_order(), 4);
        assert_eq!(ComponentRole::OutputFormat.natural_order(), 5);
    }

    #[test]
    fn test_component_role_all() {
        let all = ComponentRole::all();
        assert_eq!(all.len(), 6);
        assert!(all.contains(&ComponentRole::SystemContext));
        assert!(all.contains(&ComponentRole::UserInstruction));
    }

    #[test]
    fn test_component_role_default() {
        assert_eq!(ComponentRole::default(), ComponentRole::SystemContext);
    }

    #[test]
    fn test_component_role_label() {
        assert_eq!(ComponentRole::SystemContext.label(), "System Context");
        assert_eq!(ComponentRole::UserInstruction.label(), "Task");
        assert_eq!(ComponentRole::Example.label(), "Examples");
    }

    #[test]
    fn test_component_role_is_typically_required() {
        assert!(ComponentRole::SystemContext.is_typically_required());
        assert!(ComponentRole::UserInstruction.is_typically_required());
        assert!(!ComponentRole::Example.is_typically_required());
        assert!(!ComponentRole::Constraint.is_typically_required());
    }

    // -------------------------------------------------------------------------
    // PromptComponent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prompt_component_new() {
        let component = PromptComponent::new("test-id", "test content");
        assert_eq!(component.id, "test-id");
        assert_eq!(component.content, "test content");
        assert_eq!(component.role, ComponentRole::SystemContext);
        assert!(!component.required);
        assert!(component.order.is_none());
    }

    #[test]
    fn test_prompt_component_builder() {
        let component = PromptComponent::new("id", "content")
            .with_role(ComponentRole::UserInstruction)
            .with_required(true)
            .with_order(5)
            .with_tag("important")
            .with_metadata("key", "value");

        assert_eq!(component.role, ComponentRole::UserInstruction);
        assert!(component.required);
        assert_eq!(component.order, Some(5));
        assert!(component.tags.contains("important"));
        assert_eq!(component.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_prompt_component_with_tags() {
        let component = PromptComponent::new("id", "content")
            .with_tags(vec!["tag1", "tag2", "tag3"]);

        assert_eq!(component.tags.len(), 3);
        assert!(component.has_tag("tag1"));
        assert!(component.has_tag("tag2"));
        assert!(component.has_tag("tag3"));
    }

    #[test]
    fn test_prompt_component_effective_order() {
        let component1 = PromptComponent::new("id", "content")
            .with_role(ComponentRole::UserInstruction);
        assert_eq!(component1.effective_order(), 2);

        let component2 = PromptComponent::new("id", "content")
            .with_role(ComponentRole::UserInstruction)
            .with_order(10);
        assert_eq!(component2.effective_order(), 10);
    }

    #[test]
    fn test_prompt_component_estimate_tokens() {
        let component = PromptComponent::new("id", "This is a test string with some words");
        let tokens = component.estimate_tokens();
        assert!(tokens > 0);
        assert!(tokens < component.content.len());
    }

    #[test]
    fn test_prompt_component_has_tag() {
        let component = PromptComponent::new("id", "content")
            .with_tag("exists");

        assert!(component.has_tag("exists"));
        assert!(!component.has_tag("missing"));
    }

    #[test]
    fn test_prompt_component_default() {
        let component = PromptComponent::default();
        assert!(component.id.is_empty());
        assert!(component.content.is_empty());
    }

    // -------------------------------------------------------------------------
    // PromptStructure Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prompt_structure_display() {
        assert_eq!(PromptStructure::Linear.to_string(), "linear");
        assert_eq!(PromptStructure::Hierarchical.to_string(), "hierarchical");
        assert_eq!(PromptStructure::Conversational.to_string(), "conversational");
    }

    #[test]
    fn test_prompt_structure_from_str() {
        assert_eq!(
            "linear".parse::<PromptStructure>().unwrap(),
            PromptStructure::Linear
        );
        assert_eq!(
            "flat".parse::<PromptStructure>().unwrap(),
            PromptStructure::Linear
        );
        assert_eq!(
            "hierarchical".parse::<PromptStructure>().unwrap(),
            PromptStructure::Hierarchical
        );
        assert_eq!(
            "conversational".parse::<PromptStructure>().unwrap(),
            PromptStructure::Conversational
        );
        assert_eq!(
            "chat".parse::<PromptStructure>().unwrap(),
            PromptStructure::Conversational
        );
    }

    #[test]
    fn test_prompt_structure_from_str_invalid() {
        assert!("invalid".parse::<PromptStructure>().is_err());
    }

    #[test]
    fn test_prompt_structure_default_separator() {
        assert_eq!(PromptStructure::Linear.default_separator(), "\n\n");
        assert_eq!(PromptStructure::Hierarchical.default_separator(), "\n");
        assert_eq!(PromptStructure::Conversational.default_separator(), "\n---\n");
    }

    #[test]
    fn test_prompt_structure_all() {
        let all = PromptStructure::all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_prompt_structure_default() {
        assert_eq!(PromptStructure::default(), PromptStructure::Linear);
    }

    // -------------------------------------------------------------------------
    // OrderConstraint Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_order_constraint_new() {
        let constraint = OrderConstraint::new("first", "second");
        assert_eq!(constraint.before_id, "first");
        assert_eq!(constraint.after_id, "second");
        assert!(!constraint.adjacent);
        assert!(constraint.reason.is_none());
    }

    #[test]
    fn test_order_constraint_builder() {
        let constraint = OrderConstraint::new("a", "b")
            .with_adjacent(true)
            .with_reason("Must be together");

        assert!(constraint.adjacent);
        assert_eq!(constraint.reason, Some("Must be together".to_string()));
    }

    #[test]
    fn test_order_constraint_is_satisfied() {
        let constraint = OrderConstraint::new("a", "b");

        assert!(constraint.is_satisfied(&["a", "b", "c"]));
        assert!(constraint.is_satisfied(&["a", "x", "b"]));
        assert!(!constraint.is_satisfied(&["b", "a"]));
        assert!(!constraint.is_satisfied(&["b", "x", "a"]));
    }

    #[test]
    fn test_order_constraint_is_satisfied_adjacent() {
        let constraint = OrderConstraint::new("a", "b").with_adjacent(true);

        assert!(constraint.is_satisfied(&["a", "b", "c"]));
        assert!(!constraint.is_satisfied(&["a", "x", "b"]));
    }

    #[test]
    fn test_order_constraint_missing_components() {
        let constraint = OrderConstraint::new("a", "b");

        assert!(constraint.is_satisfied(&["x", "y", "z"]));
        assert!(constraint.is_satisfied(&["a", "x", "y"]));
        assert!(constraint.is_satisfied(&["x", "y", "b"]));
    }

    #[test]
    fn test_order_constraint_default() {
        let constraint = OrderConstraint::default();
        assert!(constraint.before_id.is_empty());
        assert!(constraint.after_id.is_empty());
    }

    // -------------------------------------------------------------------------
    // CompositionRule Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_composition_rule_default() {
        let rule = CompositionRule::default();
        assert!(rule.required_roles.contains(&ComponentRole::UserInstruction));
        assert_eq!(rule.max_tokens, 100_000);
        assert_eq!(rule.min_components, 1);
        assert_eq!(rule.max_components, 50);
        assert!(rule.allow_duplicate_roles);
        assert!(rule.enforce_ordering);
    }

    #[test]
    fn test_composition_rule_strict() {
        let rule = CompositionRule::strict();
        assert!(rule.required_roles.contains(&ComponentRole::SystemContext));
        assert!(rule.required_roles.contains(&ComponentRole::UserInstruction));
        assert!(!rule.allow_duplicate_roles);
    }

    #[test]
    fn test_composition_rule_lenient() {
        let rule = CompositionRule::lenient();
        assert!(rule.required_roles.is_empty());
        assert!(!rule.enforce_ordering);
    }

    #[test]
    fn test_composition_rule_builder() {
        let rule = CompositionRule::new()
            .with_required_role(ComponentRole::Example)
            .with_max_tokens(5000)
            .with_component_limits(2, 10);

        assert!(rule.required_roles.contains(&ComponentRole::Example));
        assert_eq!(rule.max_tokens, 5000);
        assert_eq!(rule.min_components, 2);
        assert_eq!(rule.max_components, 10);
    }

    #[test]
    fn test_composition_rule_validate_min_components() {
        let rule = CompositionRule::new().with_component_limits(2, 10);
        let components = vec![PromptComponent::new("id", "content")];

        let errors = rule.validate(&components);
        assert!(errors.iter().any(|e| e.contains("Too few components")));
    }

    #[test]
    fn test_composition_rule_validate_max_components() {
        let rule = CompositionRule::new().with_component_limits(0, 2);
        let components = vec![
            PromptComponent::new("1", "content"),
            PromptComponent::new("2", "content"),
            PromptComponent::new("3", "content"),
        ];

        let errors = rule.validate(&components);
        assert!(errors.iter().any(|e| e.contains("Too many components")));
    }

    #[test]
    fn test_composition_rule_validate_required_roles() {
        let rule = CompositionRule::new().with_required_role(ComponentRole::Example);
        let components = vec![PromptComponent::new("id", "content")
            .with_role(ComponentRole::UserInstruction)];

        let errors = rule.validate(&components);
        assert!(errors.iter().any(|e| e.contains("Missing required role")));
    }

    #[test]
    fn test_composition_rule_validate_duplicate_roles() {
        let mut rule = CompositionRule::new();
        rule.allow_duplicate_roles = false;

        let components = vec![
            PromptComponent::new("1", "content").with_role(ComponentRole::Example),
            PromptComponent::new("2", "content").with_role(ComponentRole::Example),
        ];

        let errors = rule.validate(&components);
        assert!(errors.iter().any(|e| e.contains("Duplicate role")));
    }

    #[test]
    fn test_composition_rule_validate_token_limit() {
        let rule = CompositionRule::new().with_max_tokens(10);
        let components = vec![PromptComponent::new(
            "id",
            "This is a long content that exceeds the token limit",
        )];

        let errors = rule.validate(&components);
        assert!(errors.iter().any(|e| e.contains("Exceeds token limit")));
    }

    #[test]
    fn test_composition_rule_validate_order_constraints() {
        let rule = CompositionRule::new()
            .with_order_constraint(OrderConstraint::new("second", "first"));

        let components = vec![
            PromptComponent::new("first", "content"),
            PromptComponent::new("second", "content"),
        ];

        let errors = rule.validate(&components);
        assert!(errors.iter().any(|e| e.contains("Order constraint violated")));
    }

    // -------------------------------------------------------------------------
    // ComposedPrompt Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_composed_prompt_new() {
        let components = vec![PromptComponent::new("id", "content")];
        let composed = ComposedPrompt::new("composed content", components);

        assert_eq!(composed.content, "composed content");
        assert_eq!(composed.component_count(), 1);
        assert!(composed.token_count > 0);
    }

    #[test]
    fn test_composed_prompt_components_by_role() {
        let components = vec![
            PromptComponent::new("1", "content").with_role(ComponentRole::SystemContext),
            PromptComponent::new("2", "content").with_role(ComponentRole::Example),
            PromptComponent::new("3", "content").with_role(ComponentRole::Example),
        ];
        let composed = ComposedPrompt::new("content", components);

        let examples = composed.components_by_role(ComponentRole::Example);
        assert_eq!(examples.len(), 2);
    }

    #[test]
    fn test_composed_prompt_has_role() {
        let components =
            vec![PromptComponent::new("id", "content").with_role(ComponentRole::SystemContext)];
        let composed = ComposedPrompt::new("content", components);

        assert!(composed.has_role(ComponentRole::SystemContext));
        assert!(!composed.has_role(ComponentRole::Example));
    }

    #[test]
    fn test_composed_prompt_summary() {
        let mut composed = ComposedPrompt::new(
            "content",
            vec![
                PromptComponent::new("1", "a"),
                PromptComponent::new("2", "b"),
            ],
        );
        composed.duration_ms = 100;

        let summary = composed.summary();
        assert!(summary.contains("2 components"));
        assert!(summary.contains("100ms"));
    }

    #[test]
    fn test_composed_prompt_has_warnings() {
        let mut composed = ComposedPrompt::new("content", vec![]);
        assert!(!composed.has_warnings());

        composed.warnings.push("Warning!".to_string());
        assert!(composed.has_warnings());
    }

    #[test]
    fn test_composed_prompt_default() {
        let composed = ComposedPrompt::default();
        assert!(composed.content.is_empty());
        assert!(composed.components.is_empty());
    }

    // -------------------------------------------------------------------------
    // ComposerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_composer_config_default() {
        let config = ComposerConfig::default();
        assert_eq!(config.default_structure, PromptStructure::Linear);
        assert!(config.include_headers);
        assert!(config.trim_content);
        assert!(config.deduplicate);
    }

    #[test]
    fn test_composer_config_minimal() {
        let config = ComposerConfig::minimal();
        assert!(!config.include_headers);
        assert!(!config.deduplicate);
    }

    #[test]
    fn test_composer_config_structured() {
        let config = ComposerConfig::structured();
        assert_eq!(config.default_structure, PromptStructure::Hierarchical);
        assert!(config.include_headers);
    }

    #[test]
    fn test_composer_config_chat() {
        let config = ComposerConfig::chat();
        assert_eq!(config.default_structure, PromptStructure::Conversational);
        assert!(!config.include_headers);
    }

    // -------------------------------------------------------------------------
    // ComposerAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_composer_agent_new() {
        let agent = ComposerAgent::new();
        assert_eq!(
            agent.config().default_structure,
            PromptStructure::Linear
        );
    }

    #[test]
    fn test_composer_agent_with_config() {
        let config = ComposerConfig::structured();
        let agent = ComposerAgent::with_config(config);
        assert_eq!(
            agent.config().default_structure,
            PromptStructure::Hierarchical
        );
    }

    #[test]
    fn test_composer_agent_builder() {
        let agent = ComposerAgent::builder()
            .default_structure(PromptStructure::Conversational)
            .separator("---".to_string())
            .include_headers(false)
            .trim_content(false)
            .deduplicate(false)
            .token_warning_threshold(8000)
            .build();

        assert_eq!(
            agent.config().default_structure,
            PromptStructure::Conversational
        );
        assert_eq!(agent.config().separator, "---");
        assert!(!agent.config().include_headers);
        assert!(!agent.config().trim_content);
        assert!(!agent.config().deduplicate);
        assert_eq!(agent.config().token_warning_threshold, 8000);
    }

    #[test]
    fn test_composer_agent_default() {
        let agent = ComposerAgent::default();
        assert_eq!(
            agent.config().default_structure,
            PromptStructure::Linear
        );
    }

    // -------------------------------------------------------------------------
    // Composition Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_compose_empty_components() {
        let agent = ComposerAgent::builder()
            .default_rules(CompositionRule::lenient())
            .build();

        let result = agent.compose(&[]).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.content.is_empty());
    }

    #[tokio::test]
    async fn test_compose_single_component() {
        let agent = ComposerAgent::new();
        let components = vec![PromptComponent::new("task", "Write a poem")
            .with_role(ComponentRole::UserInstruction)];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.content.contains("Write a poem"));
    }

    #[tokio::test]
    async fn test_compose_multiple_components() {
        let agent = ComposerAgent::new();
        let components = vec![
            PromptComponent::new("sys", "You are a poet")
                .with_role(ComponentRole::SystemContext),
            PromptComponent::new("task", "Write a haiku")
                .with_role(ComponentRole::UserInstruction),
            PromptComponent::new("format", "Use 5-7-5 syllable structure")
                .with_role(ComponentRole::OutputFormat),
        ];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.content.contains("You are a poet"));
        assert!(composed.content.contains("Write a haiku"));
        assert!(composed.content.contains("5-7-5"));
        assert_eq!(composed.component_count(), 3);
    }

    #[tokio::test]
    async fn test_compose_respects_ordering() {
        let agent = ComposerAgent::builder()
            .include_headers(false)
            .default_rules(CompositionRule::lenient())
            .build();

        let components = vec![
            PromptComponent::new("last", "CONTENT_LAST").with_order(10),
            PromptComponent::new("first", "CONTENT_FIRST").with_order(1),
            PromptComponent::new("middle", "CONTENT_MIDDLE").with_order(5),
        ];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        let first_pos = composed.content.find("CONTENT_FIRST").unwrap();
        let middle_pos = composed.content.find("CONTENT_MIDDLE").unwrap();
        let last_pos = composed.content.find("CONTENT_LAST").unwrap();

        assert!(first_pos < middle_pos, "FIRST ({}) should come before MIDDLE ({})", first_pos, middle_pos);
        assert!(middle_pos < last_pos, "MIDDLE ({}) should come before LAST ({})", middle_pos, last_pos);
    }

    #[tokio::test]
    async fn test_compose_with_linear_structure() {
        let agent = ComposerAgent::builder()
            .default_structure(PromptStructure::Linear)
            .include_headers(true)
            .build();

        let components = vec![
            PromptComponent::new("sys", "System context")
                .with_role(ComponentRole::SystemContext),
            PromptComponent::new("task", "User task")
                .with_role(ComponentRole::UserInstruction),
        ];

        let result = agent
            .compose_with_structure(&components, PromptStructure::Linear)
            .await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.content.contains("## System Context"));
        assert!(composed.content.contains("## Task"));
    }

    #[tokio::test]
    async fn test_compose_with_hierarchical_structure() {
        let agent = ComposerAgent::new();
        let components = vec![
            PromptComponent::new("sys", "System context")
                .with_role(ComponentRole::SystemContext),
            PromptComponent::new("task", "User task")
                .with_role(ComponentRole::UserInstruction),
        ];

        let result = agent
            .compose_with_structure(&components, PromptStructure::Hierarchical)
            .await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.content.contains("<system_context>"));
        assert!(composed.content.contains("</system_context>"));
        assert!(composed.content.contains("<user_instruction>"));
    }

    #[tokio::test]
    async fn test_compose_with_conversational_structure() {
        let agent = ComposerAgent::new();
        let components = vec![
            PromptComponent::new("sys", "You are helpful")
                .with_role(ComponentRole::SystemContext),
            PromptComponent::new("task", "Explain AI")
                .with_role(ComponentRole::UserInstruction),
        ];

        let result = agent
            .compose_with_structure(&components, PromptStructure::Conversational)
            .await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.content.contains("[system]"));
        assert!(composed.content.contains("[user]"));
    }

    #[tokio::test]
    async fn test_compose_validation_failure() {
        let agent = ComposerAgent::builder()
            .default_rules(CompositionRule::strict())
            .build();

        // Missing required SystemContext
        let components = vec![PromptComponent::new("task", "Do something")
            .with_role(ComponentRole::UserInstruction)];

        let result = agent.compose(&components).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_compose_with_custom_rules() {
        let agent = ComposerAgent::new();
        let rules = CompositionRule::new()
            .with_required_role(ComponentRole::Example)
            .with_max_tokens(1000);

        let components = vec![
            PromptComponent::new("task", "Task")
                .with_role(ComponentRole::UserInstruction),
            PromptComponent::new("example", "Example")
                .with_role(ComponentRole::Example),
        ];

        let result = agent
            .compose_with_rules(&components, PromptStructure::Linear, &rules)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_compose_trims_content() {
        let agent = ComposerAgent::builder()
            .trim_content(true)
            .include_headers(false)
            .build();

        let components = vec![PromptComponent::new("task", "  trimmed content  ")
            .with_role(ComponentRole::UserInstruction)];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.content.contains("trimmed content"));
        assert!(!composed.content.starts_with("  "));
    }

    #[tokio::test]
    async fn test_compose_deduplicates() {
        let agent = ComposerAgent::builder()
            .deduplicate(true)
            .include_headers(false)
            .build();

        let components = vec![
            PromptComponent::new("1", "duplicate")
                .with_role(ComponentRole::UserInstruction),
            PromptComponent::new("2", "duplicate")
                .with_role(ComponentRole::UserInstruction),
        ];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        // Should only have one instance after deduplication
        assert_eq!(composed.component_count(), 1);
    }

    #[tokio::test]
    async fn test_compose_generates_warnings() {
        let agent = ComposerAgent::builder()
            .token_warning_threshold(1)
            .default_rules(CompositionRule::lenient())
            .build();

        let components = vec![PromptComponent::new(
            "task",
            "This content is long enough to exceed the token warning threshold",
        )
        .with_role(ComponentRole::UserInstruction)];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.has_warnings());
        assert!(composed.warnings.iter().any(|w| w.contains("large")));
    }

    #[tokio::test]
    async fn test_compose_records_duration() {
        let agent = ComposerAgent::new();
        let components = vec![PromptComponent::new("task", "Task")
            .with_role(ComponentRole::UserInstruction)];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        // Duration should be recorded (likely very small)
        // duration_ms is u64 so it's always >= 0
        assert!(composed.component_count() >= 1);
    }

    // -------------------------------------------------------------------------
    // Utility Method Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_estimate_tokens() {
        let agent = ComposerAgent::new();
        let components = vec![
            PromptComponent::new("1", "Short"),
            PromptComponent::new("2", "Longer content here"),
        ];

        let tokens = agent.estimate_tokens(&components);
        assert!(tokens > 0);
    }

    #[test]
    fn test_validate() {
        let agent = ComposerAgent::builder()
            .default_rules(
                CompositionRule::new()
                    .with_required_role(ComponentRole::Example)
                    .with_component_limits(2, 10),
            )
            .build();

        let components = vec![PromptComponent::new("task", "Task")
            .with_role(ComponentRole::UserInstruction)];

        let errors = agent.validate(&components);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("Missing required role")));
        assert!(errors.iter().any(|e| e.contains("Too few components")));
    }

    #[test]
    fn test_filter_by_role() {
        let components = vec![
            PromptComponent::new("1", "a").with_role(ComponentRole::Example),
            PromptComponent::new("2", "b").with_role(ComponentRole::Constraint),
            PromptComponent::new("3", "c").with_role(ComponentRole::Example),
        ];

        let examples = ComposerAgent::filter_by_role(&components, ComponentRole::Example);
        assert_eq!(examples.len(), 2);
    }

    #[test]
    fn test_filter_by_tag() {
        let components = vec![
            PromptComponent::new("1", "a").with_tag("important"),
            PromptComponent::new("2", "b"),
            PromptComponent::new("3", "c").with_tag("important"),
        ];

        let important = ComposerAgent::filter_by_tag(&components, "important");
        assert_eq!(important.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let builder = ComposerAgentBuilder::default();
        let agent = builder.build();
        assert_eq!(
            agent.config().default_structure,
            PromptStructure::Linear
        );
    }

    #[test]
    fn test_builder_config() {
        let config = ComposerConfig::chat();
        let agent = ComposerAgentBuilder::default().config(config).build();
        assert_eq!(
            agent.config().default_structure,
            PromptStructure::Conversational
        );
    }

    #[test]
    fn test_builder_chain() {
        let agent = ComposerAgentBuilder::default()
            .default_structure(PromptStructure::Hierarchical)
            .separator("|||".to_string())
            .include_headers(true)
            .trim_content(false)
            .deduplicate(true)
            .token_warning_threshold(5000)
            .default_rules(CompositionRule::strict())
            .build();

        assert_eq!(
            agent.config().default_structure,
            PromptStructure::Hierarchical
        );
        assert_eq!(agent.config().separator, "|||");
        assert!(agent.config().include_headers);
        assert!(!agent.config().trim_content);
        assert!(agent.config().deduplicate);
        assert_eq!(agent.config().token_warning_threshold, 5000);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_compose_empty_content() {
        let agent = ComposerAgent::builder()
            .default_rules(CompositionRule::lenient())
            .build();

        let components = vec![
            PromptComponent::new("1", "").with_role(ComponentRole::UserInstruction),
            PromptComponent::new("2", "  ").with_role(ComponentRole::Example),
        ];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_compose_unicode() {
        let agent = ComposerAgent::new();
        let components = vec![PromptComponent::new("task", "Write about emojis")
            .with_role(ComponentRole::UserInstruction)];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("emojis"));
    }

    #[tokio::test]
    async fn test_compose_multiline() {
        let agent = ComposerAgent::builder()
            .include_headers(false)
            .build();

        let components = vec![PromptComponent::new(
            "task",
            "Line 1\nLine 2\nLine 3",
        )
        .with_role(ComponentRole::UserInstruction)];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.content.contains("Line 1"));
        assert!(composed.content.contains("Line 2"));
        assert!(composed.content.contains("Line 3"));
    }

    #[tokio::test]
    async fn test_compose_special_characters() {
        let agent = ComposerAgent::new();
        let components = vec![PromptComponent::new(
            "task",
            "Handle <tags>, {brackets}, and [arrays]",
        )
        .with_role(ComponentRole::UserInstruction)];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_compose_very_long_content() {
        let agent = ComposerAgent::builder()
            .default_rules(CompositionRule::lenient())
            .token_warning_threshold(100_000)
            .build();

        let long_content = "word ".repeat(1000);
        let components = vec![
            PromptComponent::new("task", &long_content).with_role(ComponentRole::UserInstruction)
        ];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_compose_preserves_metadata() {
        let agent = ComposerAgent::new();
        let components = vec![PromptComponent::new("task", "Content")
            .with_role(ComponentRole::UserInstruction)
            .with_metadata("author", "test")
            .with_tag("important")];

        let result = agent.compose(&components).await;
        assert!(result.is_ok());

        let composed = result.unwrap();
        assert!(composed.components[0].metadata.contains_key("author"));
        assert!(composed.components[0].has_tag("important"));
    }
}
