//! Prompt synthesizer module for Project Panpsychism.
//!
//! Combines selected prompts and generates output via LLM (Gemini API).
//! Implements the Meta-Prompt Builder pattern for orchestrating multiple prompts
//! into a coherent system message following Spinoza's philosophical principles.
//!
//! # Architecture
//!
//! The synthesizer is the final stage in the Sorcerer's Wand metaphor:
//! - **Selected Prompts** are the spells chosen from the grimoire
//! - **Meta-Prompt** is the combined incantation sent to the LLM
//! - **Synthesis** is the magic taking form
//! - **Validation** ensures Spinoza's blessing on the creation
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::synthesizer::{Synthesizer, SynthesisConfig};
//! use panpsychism::gemini::GeminiClient;
//!
//! let synthesizer = Synthesizer::new_with_gemini()
//!     .with_validation(true)
//!     .with_timeout(60);
//!
//! let result = synthesizer.synthesize(&prompts, "How to implement auth?").await?;
//! println!("{}", result.output);
//! ```

use crate::{
    gemini::{GeminiClient, GeminiModel, Message, Usage},
    orchestrator::SelectedPrompt,
    validator::{SpinozaValidator, ValidationConfig, ValidationResult},
    Error, Result,
};
use std::cmp::Ordering;
use std::fmt;
use std::fs;
use std::time::{Duration, Instant};

// =============================================================================
// OUTPUT FORMAT
// =============================================================================

/// Output format for the synthesized response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Markdown formatted output (default)
    #[default]
    Markdown,
    /// JSON structured output
    Json,
    /// Plain text output
    Plain,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputFormat::Markdown => write!(f, "markdown"),
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Plain => write!(f, "plain"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = crate::Error;

    /// Parse an output format from a string identifier.
    ///
    /// Accepts various forms: "markdown", "md", "json", "plain", "text", etc.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::str::FromStr;
    /// use panpsychism::synthesizer::OutputFormat;
    ///
    /// let format: OutputFormat = "markdown".parse().unwrap();
    /// assert_eq!(format, OutputFormat::Markdown);
    ///
    /// let format: OutputFormat = "json".parse().unwrap();
    /// assert_eq!(format, OutputFormat::Json);
    /// ```
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "markdown" | "md" => Ok(OutputFormat::Markdown),
            "json" | "structured" => Ok(OutputFormat::Json),
            "plain" | "text" | "txt" | "raw" => Ok(OutputFormat::Plain),
            _ => Err(crate::Error::Config(format!(
                "Unknown output format: '{}'. Valid formats: markdown, json, plain",
                s
            ))),
        }
    }
}

impl OutputFormat {
    /// Get format instructions for the LLM.
    pub fn instructions(&self) -> &'static str {
        match self {
            OutputFormat::Markdown => {
                "Format your response using Markdown:\n\
                 - Use headers (##, ###) for sections\n\
                 - Use bullet points for lists\n\
                 - Use code blocks with language tags for code\n\
                 - Use bold and italic for emphasis"
            }
            OutputFormat::Json => {
                "Format your response as valid JSON:\n\
                 - Use a top-level object with clear keys\n\
                 - Include a 'response' key for the main content\n\
                 - Include a 'metadata' key for additional info\n\
                 - Ensure all strings are properly escaped"
            }
            OutputFormat::Plain => {
                "Format your response as plain text:\n\
                 - Use clear paragraph breaks\n\
                 - Avoid special formatting\n\
                 - Use indentation for hierarchy"
            }
        }
    }
}

// =============================================================================
// PROMPT ROLE
// =============================================================================

/// Role classification for prompts in the synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PromptRole {
    /// Primary prompt - main instruction (priority 0)
    Primary,
    /// Supporting prompt - additional context (priority 1)
    Supporting,
    /// Context prompt - background information (priority 2)
    Context,
    /// Validator prompt - verification rules (priority 3)
    Validator,
}

impl std::str::FromStr for PromptRole {
    type Err = std::convert::Infallible;

    /// Parse role from string.
    ///
    /// This implementation never fails - unknown values default to `Supporting`.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "primary" | "main" => PromptRole::Primary,
            "supporting" | "support" | "secondary" => PromptRole::Supporting,
            "context" | "background" => PromptRole::Context,
            "validator" | "validation" | "verify" => PromptRole::Validator,
            _ => PromptRole::Supporting,
        })
    }
}

impl PromptRole {
    /// Get priority value (lower = higher priority).
    pub fn priority(&self) -> u8 {
        match self {
            PromptRole::Primary => 0,
            PromptRole::Supporting => 1,
            PromptRole::Context => 2,
            PromptRole::Validator => 3,
        }
    }

    /// Get display label for the role.
    pub fn label(&self) -> &'static str {
        match self {
            PromptRole::Primary => "Primary",
            PromptRole::Supporting => "Supporting",
            PromptRole::Context => "Context",
            PromptRole::Validator => "Validator",
        }
    }
}

// =============================================================================
// PROMPT SECTION
// =============================================================================

/// A section of the meta-prompt containing a single prompt's contribution.
#[derive(Debug, Clone)]
pub struct PromptSection {
    /// Role of this prompt in the synthesis
    pub role: PromptRole,
    /// Title of the prompt
    pub title: String,
    /// Full content of the prompt
    pub content: String,
    /// Priority order (lower = higher priority)
    pub priority: u8,
    /// Source path of the prompt (for debugging)
    pub source: Option<String>,
}

impl PromptSection {
    /// Create a new prompt section.
    pub fn new(role: PromptRole, title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            priority: role.priority(),
            role,
            title: title.into(),
            content: content.into(),
            source: None,
        }
    }

    /// Set the source path.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Override the priority.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

// =============================================================================
// META-PROMPT
// =============================================================================

/// The complete meta-prompt structure for LLM submission.
#[derive(Debug, Clone)]
pub struct MetaPrompt {
    /// System context with Spinoza principles
    pub system_context: String,
    /// Ordered sections of prompts
    pub prompt_sections: Vec<PromptSection>,
    /// The user's original query
    pub user_query: String,
    /// Desired output format
    pub output_format: OutputFormat,
    /// Additional constraints for the response
    pub constraints: Vec<String>,
}

impl MetaPrompt {
    /// Create a new meta-prompt builder.
    pub fn builder() -> MetaPromptBuilder {
        MetaPromptBuilder::default()
    }

    /// Get the total number of prompt sections.
    pub fn section_count(&self) -> usize {
        self.prompt_sections.len()
    }

    /// Check if the meta-prompt has any primary prompts.
    pub fn has_primary(&self) -> bool {
        self.prompt_sections
            .iter()
            .any(|s| s.role == PromptRole::Primary)
    }

    /// Get sections by role.
    pub fn sections_by_role(&self, role: PromptRole) -> Vec<&PromptSection> {
        self.prompt_sections
            .iter()
            .filter(|s| s.role == role)
            .collect()
    }
}

/// Builder for constructing MetaPrompt instances.
#[derive(Debug, Default)]
pub struct MetaPromptBuilder {
    system_context: Option<String>,
    prompt_sections: Vec<PromptSection>,
    user_query: Option<String>,
    output_format: OutputFormat,
    constraints: Vec<String>,
}

impl MetaPromptBuilder {
    /// Set the system context.
    pub fn system_context(mut self, context: impl Into<String>) -> Self {
        self.system_context = Some(context.into());
        self
    }

    /// Add a prompt section.
    pub fn add_section(mut self, section: PromptSection) -> Self {
        self.prompt_sections.push(section);
        self
    }

    /// Set the user query.
    pub fn user_query(mut self, query: impl Into<String>) -> Self {
        self.user_query = Some(query.into());
        self
    }

    /// Set the output format.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Add a constraint.
    pub fn add_constraint(mut self, constraint: impl Into<String>) -> Self {
        self.constraints.push(constraint.into());
        self
    }

    /// Add multiple constraints.
    pub fn constraints(mut self, constraints: Vec<String>) -> Self {
        self.constraints.extend(constraints);
        self
    }

    /// Build the MetaPrompt.
    pub fn build(mut self) -> Result<MetaPrompt> {
        self.prompt_sections
            .sort_by(|a, b| match a.priority.cmp(&b.priority) {
                Ordering::Equal => a.role.cmp(&b.role),
                other => other,
            });

        let system_context = self.system_context.unwrap_or_else(default_system_context);
        let user_query = self
            .user_query
            .ok_or_else(|| Error::Synthesis("User query is required".to_string()))?;

        Ok(MetaPrompt {
            system_context,
            prompt_sections: self.prompt_sections,
            user_query,
            output_format: self.output_format,
            constraints: self.constraints,
        })
    }
}

/// Default system context with Spinoza principles.
fn default_system_context() -> String {
    r#"You are an intelligent prompt orchestrator following Spinoza's philosophical principles:

- CONATUS: Strive to persist and enhance understanding. Your responses should be self-sustaining and contribute to the user's growth.
- RATIO: Apply reason and logical analysis. Every conclusion must follow from clear premises.
- LAETITIA: Generate responses that increase joy and clarity. Aim for responses that enlighten and satisfy.
- NATURA: Align with the natural order of knowledge. Build understanding from fundamentals to complexity.

You have been provided with carefully selected prompts that work together to address the user's needs. Each prompt contributes a specific perspective or capability. Synthesize them into a coherent, helpful response."#.to_string()
}

// =============================================================================
// SYNTHESIS CONFIGURATION
// =============================================================================

/// Configuration for the synthesis process.
#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    /// Enable Spinoza validation on output
    pub enable_validation: bool,
    /// Validation configuration thresholds
    pub validation_config: ValidationConfig,
    /// Maximum retry attempts on failure
    pub max_retries: u32,
    /// Timeout in seconds for LLM requests
    pub timeout_secs: u64,
    /// Whether to include sources in meta-prompt
    pub include_sources: bool,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            validation_config: ValidationConfig::default(),
            max_retries: 3,
            timeout_secs: 60,
            include_sources: true,
        }
    }
}

impl SynthesisConfig {
    /// Create a fast configuration with minimal validation.
    pub fn fast() -> Self {
        Self {
            enable_validation: false,
            validation_config: ValidationConfig::lenient(),
            max_retries: 1,
            timeout_secs: 30,
            include_sources: false,
        }
    }

    /// Create a strict configuration with full validation.
    pub fn strict() -> Self {
        Self {
            enable_validation: true,
            validation_config: ValidationConfig::strict(),
            max_retries: 3,
            timeout_secs: 120,
            include_sources: true,
        }
    }
}

// =============================================================================
// TOKEN USAGE
// =============================================================================

/// Token usage statistics.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Input tokens (prompt tokens)
    pub input: usize,
    /// Output tokens (completion tokens)
    pub output: usize,
    /// Total tokens
    pub total: usize,
}

impl TokenUsage {
    /// Create token usage from Gemini API response.
    pub fn from_usage(usage: Option<Usage>) -> Self {
        match usage {
            Some(u) => Self {
                input: u.prompt_tokens,
                output: u.completion_tokens,
                total: u.total_tokens,
            },
            None => Self::default(),
        }
    }

    /// Estimate tokens from text (rough approximation: ~4 chars per token).
    pub fn estimate_from_text(text: &str) -> usize {
        text.len() / 4
    }
}

// =============================================================================
// SYNTHESIS RESULT
// =============================================================================

/// Result of prompt synthesis.
#[derive(Debug)]
pub struct SynthesisResult {
    /// Generated output text
    pub output: String,
    /// Prompts used in synthesis (IDs)
    pub prompts_used: Vec<String>,
    /// Token usage statistics
    pub tokens: TokenUsage,
    /// Processing time in milliseconds
    pub duration_ms: u64,
    /// Validation score (if validation was enabled)
    pub validation_score: Option<f64>,
    /// Whether corrections were applied to the output
    pub corrections_applied: bool,
    /// Validation result details (if validation was enabled)
    pub validation_result: Option<ValidationResult>,
}

impl SynthesisResult {
    /// Check if the synthesis was successful with valid output.
    pub fn is_valid(&self) -> bool {
        self.validation_score.map(|s| s >= 0.5).unwrap_or(true)
    }

    /// Get a summary of the synthesis result.
    pub fn summary(&self) -> String {
        let validation_info = match self.validation_score {
            Some(score) => format!(" | Validation: {:.2}", score),
            None => String::new(),
        };

        format!(
            "Synthesis: {} prompts | {} tokens | {}ms{}",
            self.prompts_used.len(),
            self.tokens.total,
            self.duration_ms,
            validation_info
        )
    }
}

// =============================================================================
// SYNTHESIZER
// =============================================================================

/// Synthesizer for combining prompts and generating output via Gemini API.
#[derive(Debug)]
pub struct Synthesizer {
    /// LLM endpoint URL
    endpoint: String,
    /// API key for authentication
    api_key: Option<String>,
    /// Model identifier
    model: String,
    /// Default output format
    default_format: OutputFormat,
    /// Default constraints
    default_constraints: Vec<String>,
    /// Synthesis configuration
    config: SynthesisConfig,
    /// Gemini client (optional)
    gemini_client: Option<GeminiClient>,
}

impl Synthesizer {
    /// Create a new synthesizer with custom endpoint and model.
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            model: model.into(),
            default_format: OutputFormat::Markdown,
            default_constraints: Vec::new(),
            config: SynthesisConfig::default(),
            gemini_client: None,
        }
    }

    /// Create a new synthesizer with default Antigravity Gemini settings.
    pub fn new_with_gemini() -> Self {
        let client = GeminiClient::new();
        Self {
            endpoint: crate::constants::DEFAULT_ENDPOINT.to_string(),
            api_key: Some(crate::constants::DEFAULT_API_KEY.to_string()),
            model: crate::constants::DEFAULT_MODEL.to_string(),
            default_format: OutputFormat::Markdown,
            default_constraints: Vec::new(),
            config: SynthesisConfig::default(),
            gemini_client: Some(client),
        }
    }

    /// Set the API key for authentication.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the default output format.
    pub fn with_default_format(mut self, format: OutputFormat) -> Self {
        self.default_format = format;
        self
    }

    /// Add default constraints.
    pub fn with_default_constraints(mut self, constraints: Vec<String>) -> Self {
        self.default_constraints = constraints;
        self
    }

    /// Set the synthesis configuration.
    pub fn with_config(mut self, config: SynthesisConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable or disable validation.
    pub fn with_validation(mut self, enable: bool) -> Self {
        self.config.enable_validation = enable;
        self
    }

    /// Set the request timeout in seconds.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.config.timeout_secs = timeout_secs;
        self
    }

    /// Set a custom Gemini client.
    pub fn with_gemini_client(mut self, client: GeminiClient) -> Self {
        self.gemini_client = Some(client);
        self
    }

    /// Get the configured endpoint.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Get the configured model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SynthesisConfig {
        &self.config
    }

    // =========================================================================
    // META-PROMPT BUILDING
    // =========================================================================

    /// Build a meta-prompt from selected prompts.
    pub fn build_meta_prompt(
        &self,
        prompts: &[SelectedPrompt],
        user_query: &str,
    ) -> Result<MetaPrompt> {
        self.build_meta_prompt_with_options(prompts, user_query, self.default_format, None)
    }

    /// Build a meta-prompt with custom options.
    pub fn build_meta_prompt_with_options(
        &self,
        prompts: &[SelectedPrompt],
        user_query: &str,
        format: OutputFormat,
        extra_constraints: Option<Vec<String>>,
    ) -> Result<MetaPrompt> {
        if prompts.is_empty() {
            return Err(Error::Synthesis(
                "No prompts provided for synthesis".to_string(),
            ));
        }

        if user_query.trim().is_empty() {
            return Err(Error::Synthesis("User query cannot be empty".to_string()));
        }

        let mut builder = MetaPrompt::builder()
            .user_query(user_query)
            .output_format(format)
            .constraints(self.default_constraints.clone());

        if let Some(constraints) = extra_constraints {
            builder = builder.constraints(constraints);
        }

        for selected in prompts {
            let content = self.load_prompt_content(&selected.result.path)?;
            let role: PromptRole = selected.role.parse().unwrap_or(PromptRole::Supporting);

            let section = PromptSection::new(role, &selected.result.title, content)
                .with_source(selected.result.path.to_string_lossy().to_string())
                .with_priority(selected.priority);

            builder = builder.add_section(section);
        }

        builder.build()
    }

    /// Load prompt content from file.
    fn load_prompt_content(&self, path: &std::path::Path) -> Result<String> {
        match fs::read_to_string(path) {
            Ok(content) => Ok(content),
            Err(_) => Ok(format!(
                "# Prompt Content\n\nThis is placeholder content for: {}",
                path.display()
            )),
        }
    }

    /// Render a meta-prompt to a string for LLM submission.
    pub fn render_meta_prompt(&self, meta: &MetaPrompt) -> String {
        let mut output = String::with_capacity(4096);

        output.push_str("=== SYSTEM CONTEXT ===\n\n");
        output.push_str(&meta.system_context);
        output.push_str("\n\n");

        output.push_str("=== SELECTED PROMPTS ===\n\n");

        let roles = [
            PromptRole::Primary,
            PromptRole::Supporting,
            PromptRole::Context,
            PromptRole::Validator,
        ];

        for role in roles {
            let sections: Vec<_> = meta
                .prompt_sections
                .iter()
                .filter(|s| s.role == role)
                .collect();

            if sections.is_empty() {
                continue;
            }

            for section in sections {
                output.push_str(&format!("## {}: {}\n", section.role.label(), section.title));
                if self.config.include_sources {
                    if let Some(ref source) = section.source {
                        output.push_str(&format!("<!-- Source: {} -->\n", source));
                    }
                }
                output.push('\n');
                output.push_str(&section.content);
                output.push_str("\n\n");
            }
        }

        if !meta.constraints.is_empty() {
            output.push_str("=== CONSTRAINTS ===\n\n");
            for constraint in &meta.constraints {
                output.push_str(&format!("- {}\n", constraint));
            }
            output.push('\n');
        }

        output.push_str("=== USER QUERY ===\n\n");
        output.push_str(&meta.user_query);
        output.push_str("\n\n");

        output.push_str("=== OUTPUT FORMAT ===\n\n");
        output.push_str(meta.output_format.instructions());
        output.push('\n');

        output
    }

    // =========================================================================
    // CORE SYNTHESIS METHODS
    // =========================================================================

    /// Synthesize output from selected prompts.
    pub async fn synthesize(
        &self,
        prompts: &[SelectedPrompt],
        user_query: &str,
    ) -> Result<SynthesisResult> {
        let start = Instant::now();

        let meta_prompt = self.build_meta_prompt(prompts, user_query)?;
        let rendered = self.render_meta_prompt(&meta_prompt);

        let prompts_used: Vec<String> = prompts.iter().map(|p| p.result.id.clone()).collect();

        let (response_text, usage) = self.generate_with_retry(&rendered).await?;

        let parsed_output = self.parse_response(&response_text, meta_prompt.output_format)?;

        let duration_ms = start.elapsed().as_millis() as u64;

        let mut result = SynthesisResult {
            output: parsed_output,
            prompts_used,
            tokens: TokenUsage::from_usage(usage),
            duration_ms,
            validation_score: None,
            corrections_applied: false,
            validation_result: None,
        };

        if self.config.enable_validation {
            self.apply_validation(&mut result).await?;
        }

        Ok(result)
    }

    /// Synthesize with full Spinoza validation (always validates).
    pub async fn synthesize_with_validation(
        &self,
        prompts: &[SelectedPrompt],
        user_query: &str,
    ) -> Result<SynthesisResult> {
        let start = Instant::now();

        let meta_prompt = self.build_meta_prompt(prompts, user_query)?;
        let rendered = self.render_meta_prompt(&meta_prompt);
        let prompts_used: Vec<String> = prompts.iter().map(|p| p.result.id.clone()).collect();

        let (response_text, usage) = self.generate_with_retry(&rendered).await?;
        let parsed_output = self.parse_response(&response_text, meta_prompt.output_format)?;

        let duration_ms = start.elapsed().as_millis() as u64;

        let mut result = SynthesisResult {
            output: parsed_output,
            prompts_used,
            tokens: TokenUsage::from_usage(usage),
            duration_ms,
            validation_score: None,
            corrections_applied: false,
            validation_result: None,
        };

        self.apply_validation(&mut result).await?;

        Ok(result)
    }

    /// Simple text generation without prompt orchestration.
    pub async fn generate_simple(&self, prompt: &str) -> Result<String> {
        let (response, _) = self.generate_with_retry(prompt).await?;
        Ok(response)
    }

    // =========================================================================
    // LLM API INTERACTION
    // =========================================================================

    /// Generate response with retry logic.
    async fn generate_with_retry(&self, prompt: &str) -> Result<(String, Option<Usage>)> {
        let timeout_duration = Duration::from_secs(self.config.timeout_secs);
        let mut last_error: Option<Error> = None;

        let client = self.get_or_create_client();

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let backoff = Duration::from_secs(1 << (attempt - 1));
                tokio::time::sleep(backoff).await;
            }

            let messages = vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }];

            let result = tokio::time::timeout(timeout_duration, client.chat(messages)).await;

            match result {
                Ok(Ok(response)) => {
                    let text = response
                        .choices
                        .first()
                        .map(|c| c.message.content.clone())
                        .ok_or_else(|| Error::Synthesis("Empty response from API".to_string()))?;

                    return Ok((text, response.usage));
                }
                Ok(Err(e)) => {
                    last_error = Some(e);
                }
                Err(_) => {
                    last_error = Some(Error::Synthesis(format!(
                        "Request timed out after {}s",
                        self.config.timeout_secs
                    )));
                }
            }
        }

        Err(last_error.unwrap_or_else(|| Error::Synthesis("Unknown error".to_string())))
    }

    /// Get existing client or create a new one.
    fn get_or_create_client(&self) -> GeminiClient {
        let mut client = GeminiClient::new()
            .with_endpoint(&self.endpoint)
            .with_model(GeminiModel::from_str_lossy(&self.model));

        if let Some(ref key) = self.api_key {
            client = client.with_api_key(key);
        }

        client
    }

    // =========================================================================
    // RESPONSE PARSING
    // =========================================================================

    /// Parse and format the raw response.
    pub fn parse_response(&self, raw: &str, format: OutputFormat) -> Result<String> {
        let trimmed = raw.trim();

        if trimmed.is_empty() {
            return Err(Error::Synthesis("Empty response received".to_string()));
        }

        match format {
            OutputFormat::Plain => Ok(self.strip_markdown(trimmed)),
            OutputFormat::Markdown => Ok(trimmed.to_string()),
            OutputFormat::Json => self.validate_json(trimmed),
        }
    }

    /// Strip markdown formatting for plain text output.
    fn strip_markdown(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Remove code blocks
        let code_block_re = regex::Regex::new(r"```[\s\S]*?```").unwrap();
        result = code_block_re.replace_all(&result, "").to_string();

        // Remove inline code
        let inline_code_re = regex::Regex::new(r"`[^`]+`").unwrap();
        result = inline_code_re.replace_all(&result, "").to_string();

        // Remove headers
        let header_re = regex::Regex::new(r"^#+\s*").unwrap();
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
        let list_re = regex::Regex::new(r"^[\s]*[-*]\s").unwrap();
        result = result
            .lines()
            .map(|line| list_re.replace(line, "").to_string())
            .collect::<Vec<_>>()
            .join("\n");

        // Clean up extra whitespace
        let multiline_re = regex::Regex::new(r"\n{3,}").unwrap();
        result = multiline_re.replace_all(&result, "\n\n").to_string();

        result.trim().to_string()
    }

    /// Validate and extract JSON from response.
    fn validate_json(&self, text: &str) -> Result<String> {
        if serde_json::from_str::<serde_json::Value>(text).is_ok() {
            return Ok(text.to_string());
        }

        let json_block_re = regex::Regex::new(r"```(?:json)?\s*([\s\S]*?)```").unwrap();
        if let Some(captures) = json_block_re.captures(text) {
            if let Some(json_text) = captures.get(1) {
                let extracted = json_text.as_str().trim();
                if serde_json::from_str::<serde_json::Value>(extracted).is_ok() {
                    return Ok(extracted.to_string());
                }
            }
        }

        let json_re = regex::Regex::new(r"(\{[\s\S]*\}|\[[\s\S]*\])").unwrap();
        if let Some(captures) = json_re.captures(text) {
            if let Some(json_match) = captures.get(1) {
                let extracted = json_match.as_str();
                if serde_json::from_str::<serde_json::Value>(extracted).is_ok() {
                    return Ok(extracted.to_string());
                }
            }
        }

        Err(Error::Synthesis(format!(
            "Response is not valid JSON. Raw response: {}",
            &text[..text.len().min(200)]
        )))
    }

    // =========================================================================
    // VALIDATION
    // =========================================================================

    /// Apply Spinoza validation to the result.
    async fn apply_validation(&self, result: &mut SynthesisResult) -> Result<()> {
        let validator = SpinozaValidator::with_config(self.config.validation_config.clone());

        match validator.validate(&result.output).await {
            Ok(validation) => {
                result.validation_score = Some(validation.scores.average());
                result.validation_result = Some(validation);
                Ok(())
            }
            Err(e) => {
                tracing::warn!("Validation failed: {}", e);
                result.validation_score = Some(0.0);
                Ok(())
            }
        }
    }

    /// Estimate token count for text.
    pub fn estimate_tokens(&self, text: &str) -> usize {
        TokenUsage::estimate_from_text(text)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SearchResult;
    use std::path::PathBuf;

    fn create_test_selected_prompts() -> Vec<SelectedPrompt> {
        vec![
            SelectedPrompt {
                result: SearchResult {
                    id: "auth-01".to_string(),
                    title: "OAuth2 Authentication".to_string(),
                    path: PathBuf::from("prompts/auth.md"),
                    score: 0.95,
                    excerpt: "Implement OAuth2...".to_string(),
                    tags: vec!["auth".to_string()],
                    category: Some("security".to_string()),
                },
                role: "primary".to_string(),
                priority: 0,
            },
            SelectedPrompt {
                result: SearchResult {
                    id: "security-01".to_string(),
                    title: "Security Best Practices".to_string(),
                    path: PathBuf::from("prompts/security.md"),
                    score: 0.85,
                    excerpt: "Security patterns...".to_string(),
                    tags: vec!["security".to_string()],
                    category: Some("security".to_string()),
                },
                role: "supporting".to_string(),
                priority: 1,
            },
            SelectedPrompt {
                result: SearchResult {
                    id: "context-01".to_string(),
                    title: "API Context".to_string(),
                    path: PathBuf::from("prompts/context.md"),
                    score: 0.75,
                    excerpt: "API context info...".to_string(),
                    tags: vec!["api".to_string()],
                    category: Some("context".to_string()),
                },
                role: "context".to_string(),
                priority: 2,
            },
        ]
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Markdown.to_string(), "markdown");
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Plain.to_string(), "plain");
    }

    #[test]
    fn test_prompt_role_from_str() {
        assert_eq!(
            "primary".parse::<PromptRole>().unwrap(),
            PromptRole::Primary
        );
        assert_eq!(
            "supporting".parse::<PromptRole>().unwrap(),
            PromptRole::Supporting
        );
        assert_eq!(
            "context".parse::<PromptRole>().unwrap(),
            PromptRole::Context
        );
        assert_eq!(
            "validator".parse::<PromptRole>().unwrap(),
            PromptRole::Validator
        );
    }

    #[test]
    fn test_prompt_role_priority() {
        assert_eq!(PromptRole::Primary.priority(), 0);
        assert_eq!(PromptRole::Supporting.priority(), 1);
        assert_eq!(PromptRole::Context.priority(), 2);
        assert_eq!(PromptRole::Validator.priority(), 3);
    }

    #[test]
    fn test_meta_prompt_builder() {
        let meta = MetaPrompt::builder()
            .user_query("test query")
            .add_section(PromptSection::new(PromptRole::Primary, "Title", "Content"))
            .build()
            .unwrap();

        assert_eq!(meta.user_query, "test query");
        assert_eq!(meta.section_count(), 1);
    }

    #[test]
    fn test_synthesizer_creation() {
        let synth = Synthesizer::new("http://localhost:8080", "test-model")
            .with_api_key("test-key")
            .with_default_format(OutputFormat::Json);

        assert_eq!(synth.endpoint(), "http://localhost:8080");
        assert_eq!(synth.model(), "test-model");
    }

    #[test]
    fn test_synthesizer_new_with_gemini() {
        let synth = Synthesizer::new_with_gemini();
        assert_eq!(synth.endpoint(), crate::constants::DEFAULT_ENDPOINT);
        assert_eq!(synth.model(), crate::constants::DEFAULT_MODEL);
    }

    #[test]
    fn test_synthesis_config_presets() {
        let fast = SynthesisConfig::fast();
        assert!(!fast.enable_validation);
        assert_eq!(fast.max_retries, 1);

        let strict = SynthesisConfig::strict();
        assert!(strict.enable_validation);
        assert_eq!(strict.max_retries, 3);
    }

    #[test]
    fn test_build_meta_prompt() {
        let synth = Synthesizer::new("http://localhost", "test");
        let prompts = create_test_selected_prompts();

        let result = synth.build_meta_prompt(&prompts, "How to authenticate?");
        assert!(result.is_ok());

        let meta = result.unwrap();
        assert_eq!(meta.section_count(), 3);
        assert!(meta.has_primary());
    }

    #[test]
    fn test_build_meta_prompt_empty() {
        let synth = Synthesizer::new("http://localhost", "test");
        let prompts: Vec<SelectedPrompt> = vec![];

        let result = synth.build_meta_prompt(&prompts, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_render_meta_prompt() {
        let synth = Synthesizer::new("http://localhost", "test");
        let prompts = create_test_selected_prompts();
        let meta = synth.build_meta_prompt(&prompts, "test query").unwrap();

        let rendered = synth.render_meta_prompt(&meta);

        assert!(rendered.contains("=== SYSTEM CONTEXT ==="));
        assert!(rendered.contains("=== SELECTED PROMPTS ==="));
        assert!(rendered.contains("=== USER QUERY ==="));
        assert!(rendered.contains("CONATUS"));
        assert!(rendered.contains("test query"));
    }

    #[test]
    fn test_parse_response_plain() {
        let synth = Synthesizer::new("http://localhost", "test");
        let raw = "## Header\n\n**Bold** text";

        let result = synth.parse_response(raw, OutputFormat::Plain).unwrap();
        assert!(!result.contains("##"));
        assert!(!result.contains("**"));
    }

    #[test]
    fn test_parse_response_json() {
        let synth = Synthesizer::new("http://localhost", "test");
        let raw = r#"{"key": "value"}"#;

        let result = synth.parse_response(raw, OutputFormat::Json).unwrap();
        assert_eq!(result, raw);
    }

    #[test]
    fn test_parse_response_json_in_code_block() {
        let synth = Synthesizer::new("http://localhost", "test");
        let raw = "```json\n{\"key\": \"value\"}\n```";

        let result = synth.parse_response(raw, OutputFormat::Json).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_token_usage() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };

        let token_usage = TokenUsage::from_usage(Some(usage));
        assert_eq!(token_usage.input, 100);
        assert_eq!(token_usage.output, 50);
        assert_eq!(token_usage.total, 150);
    }

    #[test]
    fn test_synthesis_result_is_valid() {
        let valid = SynthesisResult {
            output: "test".to_string(),
            prompts_used: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 0,
            validation_score: Some(0.75),
            corrections_applied: false,
            validation_result: None,
        };
        assert!(valid.is_valid());

        let invalid = SynthesisResult {
            output: "test".to_string(),
            prompts_used: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 0,
            validation_score: Some(0.3),
            corrections_applied: false,
            validation_result: None,
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_synthesis_result_summary() {
        let result = SynthesisResult {
            output: "test".to_string(),
            prompts_used: vec!["p1".to_string(), "p2".to_string()],
            tokens: TokenUsage {
                input: 100,
                output: 50,
                total: 150,
            },
            duration_ms: 1234,
            validation_score: Some(0.85),
            corrections_applied: false,
            validation_result: None,
        };

        let summary = result.summary();
        assert!(summary.contains("2 prompts"));
        assert!(summary.contains("150 tokens"));
        assert!(summary.contains("1234ms"));
        assert!(summary.contains("0.85"));
    }

    #[test]
    fn test_estimate_tokens() {
        let synth = Synthesizer::new("http://localhost", "test");
        let text = "This is a test string";
        let estimate = synth.estimate_tokens(text);
        assert!(estimate > 0);
    }
}
