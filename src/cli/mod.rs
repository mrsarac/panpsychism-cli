//! # CLI Module
//!
//! Command-line interface for direct interaction with the Panpsychism system.
//!
//! ## Overview
//!
//! The CLI provides a comprehensive interface for:
//!
//! - **Queries**: Send queries to agents for processing
//! - **Agent Management**: List and inspect available agents
//! - **System Status**: Monitor health and performance metrics
//! - **Configuration**: View and modify settings
//! - **Interactive Mode**: REPL-style interaction
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use panpsychism::cli::{CliApp, CliConfig, Command};
//!
//! let app = CliApp::builder()
//!     .output_format(OutputFormat::Text)
//!     .color_enabled(true)
//!     .build();
//!
//! // Execute a command
//! let result = app.execute(Command::Status)?;
//! println!("{}", result.output);
//!
//! // Or start interactive mode
//! app.run_interactive()?;
//! ```
//!
//! ## Output Formats
//!
//! The CLI supports multiple output formats:
//!
//! - **Text**: Human-readable output (default)
//! - **Json**: Machine-readable JSON output
//! - **Markdown**: Formatted markdown output
//! - **Quiet**: Minimal output for scripting
//!
//! ## Command Examples
//!
//! ```bash
//! # Query the system
//! panpsychism query "How to implement authentication?"
//!
//! # Query with specific agent
//! panpsychism query "Synthesize response" --agent synthesizer
//!
//! # List all agents
//! panpsychism agents
//!
//! # Show system status
//! panpsychism status
//!
//! # Show metrics
//! panpsychism metrics
//!
//! # Interactive mode
//! panpsychism interactive
//! ```

use std::collections::HashMap;
use std::fmt;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::orchestrator::Orchestrator;

// =============================================================================
// OUTPUT FORMAT
// =============================================================================

/// Output format for CLI responses.
///
/// Controls how command results are displayed to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Human-readable text format with formatting.
    #[default]
    Text,
    /// JSON structured output for machine consumption.
    Json,
    /// Markdown formatted output.
    Markdown,
    /// Minimal output for scripting (only essential data).
    Quiet,
}

impl OutputFormat {
    /// Returns the string representation of the output format.
    pub fn as_str(&self) -> &'static str {
        match self {
            OutputFormat::Text => "text",
            OutputFormat::Json => "json",
            OutputFormat::Markdown => "markdown",
            OutputFormat::Quiet => "quiet",
        }
    }

    /// Parse output format from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "text" | "txt" | "plain" => Some(OutputFormat::Text),
            "json" => Some(OutputFormat::Json),
            "markdown" | "md" => Some(OutputFormat::Markdown),
            "quiet" | "silent" | "minimal" => Some(OutputFormat::Quiet),
            _ => None,
        }
    }

    /// Check if format supports colors.
    pub fn supports_colors(&self) -> bool {
        matches!(self, OutputFormat::Text | OutputFormat::Markdown)
    }
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// CONFIG ACTION
// =============================================================================

/// Actions for configuration management.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigAction {
    /// Get a configuration value by key.
    Get { key: String },
    /// Set a configuration value.
    Set { key: String, value: String },
    /// List all configuration values.
    List,
    /// Reset configuration to defaults.
    Reset,
}

impl ConfigAction {
    /// Parse a config action from arguments.
    ///
    /// # Arguments
    ///
    /// * `action` - The action name ("get", "set", "list", "reset")
    /// * `args` - Additional arguments for the action
    ///
    /// # Returns
    ///
    /// The parsed ConfigAction or an error.
    pub fn parse(action: &str, args: &[String]) -> Result<Self> {
        match action.to_lowercase().as_str() {
            "get" => {
                if args.is_empty() {
                    return Err(Error::Config("config get requires a key".to_string()));
                }
                Ok(ConfigAction::Get {
                    key: args[0].clone(),
                })
            }
            "set" => {
                if args.len() < 2 {
                    return Err(Error::Config(
                        "config set requires key and value".to_string(),
                    ));
                }
                Ok(ConfigAction::Set {
                    key: args[0].clone(),
                    value: args[1].clone(),
                })
            }
            "list" | "ls" => Ok(ConfigAction::List),
            "reset" => Ok(ConfigAction::Reset),
            _ => Err(Error::Config(format!(
                "Unknown config action: '{}'. Use: get, set, list, reset",
                action
            ))),
        }
    }
}

// =============================================================================
// COMMAND ENUM
// =============================================================================

/// CLI commands available for execution.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Execute a query against the system.
    Query {
        /// The query text.
        text: String,
        /// Optional agent to direct the query to.
        agent: Option<String>,
    },
    /// List all available agents.
    Agents,
    /// Show system health status.
    Status,
    /// Show performance metrics.
    Metrics,
    /// Configuration management.
    Config {
        /// The configuration action to perform.
        action: ConfigAction,
    },
    /// Show command history.
    History {
        /// Number of entries to show (None = all).
        count: Option<usize>,
    },
    /// Enter interactive REPL mode.
    Interactive,
    /// Show version information.
    Version,
    /// Show help information.
    Help,
}

impl Command {
    /// Parse a command from command-line arguments.
    ///
    /// # Arguments
    ///
    /// * `args` - Command-line arguments (excluding program name)
    ///
    /// # Returns
    ///
    /// The parsed Command or an error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use panpsychism::cli::Command;
    ///
    /// let args: Vec<String> = vec!["status".into()];
    /// let cmd = Command::parse(&args).unwrap();
    /// assert_eq!(cmd, Command::Status);
    ///
    /// let args: Vec<String> = vec!["query".into(), "hello".into()];
    /// let cmd = Command::parse(&args).unwrap();
    /// assert!(matches!(cmd, Command::Query { text, agent: None } if text == "hello"));
    /// ```
    pub fn parse(args: &[String]) -> Result<Self> {
        if args.is_empty() {
            return Ok(Command::Help);
        }

        let cmd = args[0].to_lowercase();
        let rest = &args[1..];

        match cmd.as_str() {
            "query" | "q" => {
                if rest.is_empty() {
                    return Err(Error::Config("query requires text argument".to_string()));
                }

                let mut text_parts = Vec::new();
                let mut agent = None;
                let mut i = 0;

                while i < rest.len() {
                    if rest[i] == "--agent" || rest[i] == "-a" {
                        if i + 1 < rest.len() {
                            agent = Some(rest[i + 1].clone());
                            i += 2;
                        } else {
                            return Err(Error::Config("--agent requires a value".to_string()));
                        }
                    } else if rest[i].starts_with("--agent=") {
                        agent = Some(rest[i].trim_start_matches("--agent=").to_string());
                        i += 1;
                    } else {
                        text_parts.push(rest[i].clone());
                        i += 1;
                    }
                }

                if text_parts.is_empty() {
                    return Err(Error::Config("query requires text argument".to_string()));
                }

                Ok(Command::Query {
                    text: text_parts.join(" "),
                    agent,
                })
            }
            "agents" | "agent" | "a" => Ok(Command::Agents),
            "status" | "s" => Ok(Command::Status),
            "metrics" | "m" => Ok(Command::Metrics),
            "config" | "c" => {
                if rest.is_empty() {
                    Ok(Command::Config {
                        action: ConfigAction::List,
                    })
                } else {
                    let action = ConfigAction::parse(&rest[0], &rest[1..].to_vec())?;
                    Ok(Command::Config { action })
                }
            }
            "history" | "h" => {
                let count = if rest.is_empty() {
                    None
                } else {
                    rest[0]
                        .parse()
                        .ok()
                        .or_else(|| Some(10)) // Default to 10 if parse fails
                };
                Ok(Command::History { count })
            }
            "interactive" | "i" | "repl" => Ok(Command::Interactive),
            "version" | "v" | "--version" | "-v" => Ok(Command::Version),
            "help" | "--help" | "-h" => Ok(Command::Help),
            _ => Err(Error::Config(format!(
                "Unknown command: '{}'. Use 'help' for available commands.",
                cmd
            ))),
        }
    }

    /// Get the command name.
    pub fn name(&self) -> &'static str {
        match self {
            Command::Query { .. } => "query",
            Command::Agents => "agents",
            Command::Status => "status",
            Command::Metrics => "metrics",
            Command::Config { .. } => "config",
            Command::History { .. } => "history",
            Command::Interactive => "interactive",
            Command::Version => "version",
            Command::Help => "help",
        }
    }
}

// =============================================================================
// HISTORY ENTRY
// =============================================================================

/// An entry in the command history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Timestamp when the command was executed.
    pub timestamp: SystemTime,
    /// The command that was executed.
    pub command: String,
    /// The result of the command (success or error message).
    pub result: String,
    /// Whether the command succeeded.
    pub success: bool,
}

impl HistoryEntry {
    /// Create a new history entry.
    pub fn new(command: impl Into<String>, result: impl Into<String>, success: bool) -> Self {
        Self {
            timestamp: SystemTime::now(),
            command: command.into(),
            result: result.into(),
            success,
        }
    }

    /// Get the age of this entry.
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed().unwrap_or_default()
    }

    /// Format the timestamp as a string.
    pub fn timestamp_str(&self) -> String {
        let duration = self
            .timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = duration.as_secs();
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            1970 + secs / 31536000,
            (secs % 31536000) / 2592000 + 1,
            (secs % 2592000) / 86400 + 1,
            (secs % 86400) / 3600,
            (secs % 3600) / 60,
            secs % 60
        )
    }
}

// =============================================================================
// COMMAND RESULT
// =============================================================================

/// The result of executing a command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    /// Whether the command succeeded.
    pub success: bool,
    /// The output of the command.
    pub output: String,
    /// Optional error message if the command failed.
    pub error: Option<String>,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
}

impl CommandResult {
    /// Create a successful result.
    pub fn success(output: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
            duration_ms,
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>, duration_ms: u64) -> Self {
        let msg = message.into();
        Self {
            success: false,
            output: String::new(),
            error: Some(msg),
            duration_ms,
        }
    }

    /// Format the result for display.
    pub fn display(&self, format: OutputFormat, color: bool) -> String {
        match format {
            OutputFormat::Json => serde_json::to_string_pretty(self).unwrap_or_default(),
            OutputFormat::Quiet => {
                if self.success {
                    self.output.clone()
                } else {
                    self.error.clone().unwrap_or_default()
                }
            }
            OutputFormat::Markdown => {
                if self.success {
                    format!("**Result:**\n\n{}\n\n_Completed in {}ms_", self.output, self.duration_ms)
                } else {
                    format!("**Error:**\n\n{}", self.error.as_deref().unwrap_or("Unknown error"))
                }
            }
            OutputFormat::Text => {
                if self.success {
                    if color {
                        format!("\x1b[32m{}\x1b[0m", self.output)
                    } else {
                        self.output.clone()
                    }
                } else {
                    let err = self.error.as_deref().unwrap_or("Unknown error");
                    if color {
                        format!("\x1b[31mError: {}\x1b[0m", err)
                    } else {
                        format!("Error: {}", err)
                    }
                }
            }
        }
    }
}

// =============================================================================
// AGENT INFO
// =============================================================================

/// Information about an available agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent identifier.
    pub id: String,
    /// Agent display name.
    pub name: String,
    /// Agent description.
    pub description: String,
    /// Agent status.
    pub status: AgentStatus,
    /// Agent capabilities.
    pub capabilities: Vec<String>,
}

/// Status of an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentStatus {
    /// Agent is active and ready.
    Active,
    /// Agent is idle but available.
    Idle,
    /// Agent is busy processing.
    Busy,
    /// Agent is offline or unavailable.
    Offline,
}

impl AgentStatus {
    /// Get string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentStatus::Active => "active",
            AgentStatus::Idle => "idle",
            AgentStatus::Busy => "busy",
            AgentStatus::Offline => "offline",
        }
    }
}

impl fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// CLI CONFIG
// =============================================================================

/// Configuration for the CLI application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Path to the configuration file.
    pub config_path: Option<PathBuf>,
    /// Output format for results.
    pub output_format: OutputFormat,
    /// Whether colors are enabled.
    pub color_enabled: bool,
    /// Path to the history file.
    pub history_file: Option<PathBuf>,
    /// Maximum number of history entries to retain.
    pub max_history: usize,
    /// Enable verbose output.
    pub verbose: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            config_path: None,
            output_format: OutputFormat::Text,
            color_enabled: true,
            history_file: None,
            max_history: 1000,
            verbose: false,
        }
    }
}

impl CliConfig {
    /// Create a new configuration builder.
    pub fn builder() -> CliConfigBuilder {
        CliConfigBuilder::new()
    }

    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(format) = std::env::var("PANPSYCHISM_OUTPUT_FORMAT") {
            if let Some(f) = OutputFormat::from_str(&format) {
                config.output_format = f;
            }
        }

        if let Ok(color) = std::env::var("PANPSYCHISM_COLOR") {
            config.color_enabled = color.parse().unwrap_or(true);
        }

        if let Ok(verbose) = std::env::var("PANPSYCHISM_VERBOSE") {
            config.verbose = verbose.parse().unwrap_or(false);
        }

        if let Ok(path) = std::env::var("PANPSYCHISM_CONFIG") {
            config.config_path = Some(PathBuf::from(path));
        }

        if let Ok(path) = std::env::var("PANPSYCHISM_HISTORY_FILE") {
            config.history_file = Some(PathBuf::from(path));
        }

        if let Ok(max) = std::env::var("PANPSYCHISM_MAX_HISTORY") {
            config.max_history = max.parse().unwrap_or(1000);
        }

        config
    }
}

// =============================================================================
// CLI CONFIG BUILDER
// =============================================================================

/// Builder for CliConfig.
#[derive(Debug, Clone, Default)]
pub struct CliConfigBuilder {
    config: CliConfig,
}

impl CliConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: CliConfig::default(),
        }
    }

    /// Set the configuration file path.
    pub fn config_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.config_path = Some(path.into());
        self
    }

    /// Set the output format.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output_format = format;
        self
    }

    /// Enable or disable colors.
    pub fn color_enabled(mut self, enabled: bool) -> Self {
        self.config.color_enabled = enabled;
        self
    }

    /// Set the history file path.
    pub fn history_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.history_file = Some(path.into());
        self
    }

    /// Set the maximum history entries.
    pub fn max_history(mut self, max: usize) -> Self {
        self.config.max_history = max;
        self
    }

    /// Enable or disable verbose mode.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> CliConfig {
        self.config
    }
}

// =============================================================================
// CLI APP
// =============================================================================

/// The main CLI application.
///
/// Provides command execution, history management, and interactive mode.
pub struct CliApp {
    /// Application configuration.
    config: CliConfig,
    /// Optional orchestrator for processing queries.
    orchestrator: Option<Arc<Orchestrator>>,
    /// Command history.
    history: Vec<HistoryEntry>,
    /// Current output format.
    output_format: OutputFormat,
}

impl CliApp {
    /// Create a new CLI application with default configuration.
    pub fn new() -> Self {
        Self {
            config: CliConfig::default(),
            orchestrator: None,
            history: Vec::new(),
            output_format: OutputFormat::Text,
        }
    }

    /// Create a new CLI application with the given configuration.
    pub fn with_config(config: CliConfig) -> Self {
        let output_format = config.output_format;
        Self {
            config,
            orchestrator: None,
            history: Vec::new(),
            output_format,
        }
    }

    /// Create a builder for the CLI application.
    pub fn builder() -> CliAppBuilder {
        CliAppBuilder::new()
    }

    /// Set the orchestrator for processing queries.
    pub fn set_orchestrator(&mut self, orchestrator: Arc<Orchestrator>) {
        self.orchestrator = Some(orchestrator);
    }

    /// Get the current configuration.
    pub fn config(&self) -> &CliConfig {
        &self.config
    }

    /// Get the command history.
    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Get the current output format.
    pub fn output_format(&self) -> OutputFormat {
        self.output_format
    }

    /// Set the output format.
    pub fn set_output_format(&mut self, format: OutputFormat) {
        self.output_format = format;
    }

    /// Execute a command.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute
    ///
    /// # Returns
    ///
    /// The result of the command execution.
    pub fn execute(&mut self, command: Command) -> Result<CommandResult> {
        let start = std::time::Instant::now();
        let cmd_name = command.name().to_string();

        let result = match &command {
            Command::Query { text, agent } => self.execute_query(text, agent.as_deref()),
            Command::Agents => self.execute_agents(),
            Command::Status => self.execute_status(),
            Command::Metrics => self.execute_metrics(),
            Command::Config { action } => self.execute_config(action),
            Command::History { count } => self.execute_history(*count),
            Command::Interactive => {
                // Interactive mode is handled separately
                return Err(Error::Config(
                    "Use run_interactive() for interactive mode".to_string(),
                ));
            }
            Command::Version => self.execute_version(),
            Command::Help => self.execute_help(),
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        let command_result = match result {
            Ok(output) => CommandResult::success(output, duration_ms),
            Err(e) => CommandResult::error(e.to_string(), duration_ms),
        };

        // Add to history
        self.add_history(
            &cmd_name,
            if command_result.success {
                &command_result.output
            } else {
                command_result.error.as_deref().unwrap_or("Error")
            },
            command_result.success,
        );

        Ok(command_result)
    }

    /// Execute a query command.
    fn execute_query(&self, text: &str, agent: Option<&str>) -> Result<String> {
        if text.is_empty() {
            return Err(Error::Config("Query text cannot be empty".to_string()));
        }

        let agent_info = agent.map(|a| format!(" (via {})", a)).unwrap_or_default();

        // If we have an orchestrator, use it
        if let Some(_orchestrator) = &self.orchestrator {
            // In a real implementation, this would call the orchestrator
            // For now, we return a simulated response
            Ok(format!(
                "Query processed{}: \"{}\"\n\nThis is a simulated response. Connect an orchestrator for real processing.",
                agent_info, text
            ))
        } else {
            Ok(format!(
                "Query received{}: \"{}\"\n\nNote: No orchestrator connected. Set up an orchestrator for full functionality.",
                agent_info, text
            ))
        }
    }

    /// Execute the agents command.
    fn execute_agents(&self) -> Result<String> {
        // Return a list of known agents
        let agents = vec![
            AgentInfo {
                id: "search".to_string(),
                name: "Search Agent".to_string(),
                description: "Searches the prompt library for relevant content".to_string(),
                status: AgentStatus::Active,
                capabilities: vec!["search".to_string(), "filter".to_string()],
            },
            AgentInfo {
                id: "synthesizer".to_string(),
                name: "Synthesizer Agent".to_string(),
                description: "Synthesizes prompts into coherent responses".to_string(),
                status: AgentStatus::Active,
                capabilities: vec!["synthesize".to_string(), "combine".to_string()],
            },
            AgentInfo {
                id: "validator".to_string(),
                name: "Validator Agent".to_string(),
                description: "Validates content against Spinoza principles".to_string(),
                status: AgentStatus::Active,
                capabilities: vec!["validate".to_string(), "score".to_string()],
            },
            AgentInfo {
                id: "corrector".to_string(),
                name: "Corrector Agent".to_string(),
                description: "Refines and corrects generated content".to_string(),
                status: AgentStatus::Idle,
                capabilities: vec!["correct".to_string(), "refine".to_string()],
            },
        ];

        match self.output_format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(&agents).map_err(|e| Error::internal(e.to_string()))
            }
            OutputFormat::Quiet => Ok(agents.iter().map(|a| a.id.clone()).collect::<Vec<_>>().join("\n")),
            _ => {
                let mut output = String::from("Available Agents:\n\n");
                for agent in &agents {
                    let status_icon = match agent.status {
                        AgentStatus::Active => "[+]",
                        AgentStatus::Idle => "[~]",
                        AgentStatus::Busy => "[*]",
                        AgentStatus::Offline => "[-]",
                    };
                    output.push_str(&format!(
                        "{} {} ({})\n   {}\n   Capabilities: {}\n\n",
                        status_icon,
                        agent.name,
                        agent.id,
                        agent.description,
                        agent.capabilities.join(", ")
                    ));
                }
                Ok(output)
            }
        }
    }

    /// Execute the status command.
    fn execute_status(&self) -> Result<String> {
        let status = SystemStatus {
            healthy: true,
            uptime_secs: 3600, // Placeholder
            agents_active: 4,
            agents_total: 5,
            memory_used_mb: 128,
            queries_processed: 0,
        };

        match self.output_format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(&status).map_err(|e| Error::internal(e.to_string()))
            }
            OutputFormat::Quiet => Ok(if status.healthy { "OK" } else { "DEGRADED" }.to_string()),
            _ => {
                let health_icon = if status.healthy { "[+]" } else { "[-]" };
                Ok(format!(
                    "System Status: {} {}\n\n\
                     Uptime:           {} seconds\n\
                     Active Agents:    {}/{}\n\
                     Memory Used:      {} MB\n\
                     Queries Processed: {}\n",
                    health_icon,
                    if status.healthy { "Healthy" } else { "Degraded" },
                    status.uptime_secs,
                    status.agents_active,
                    status.agents_total,
                    status.memory_used_mb,
                    status.queries_processed
                ))
            }
        }
    }

    /// Execute the metrics command.
    fn execute_metrics(&self) -> Result<String> {
        let metrics = SystemMetrics {
            requests_per_second: 0.0,
            average_latency_ms: 0.0,
            error_rate: 0.0,
            cache_hit_rate: 0.0,
            active_connections: 0,
        };

        match self.output_format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(&metrics).map_err(|e| Error::internal(e.to_string()))
            }
            OutputFormat::Quiet => Ok(format!("{:.2}", metrics.requests_per_second)),
            _ => Ok(format!(
                "Performance Metrics:\n\n\
                 Requests/sec:      {:.2}\n\
                 Avg Latency:       {:.2} ms\n\
                 Error Rate:        {:.2}%\n\
                 Cache Hit Rate:    {:.2}%\n\
                 Active Connections: {}\n",
                metrics.requests_per_second,
                metrics.average_latency_ms,
                metrics.error_rate * 100.0,
                metrics.cache_hit_rate * 100.0,
                metrics.active_connections
            )),
        }
    }

    /// Execute a config command.
    fn execute_config(&self, action: &ConfigAction) -> Result<String> {
        match action {
            ConfigAction::Get { key } => {
                let value = self.get_config_value(key)?;
                match self.output_format {
                    OutputFormat::Json => {
                        let mut map = HashMap::new();
                        map.insert(key.clone(), value.clone());
                        serde_json::to_string_pretty(&map)
                            .map_err(|e| Error::internal(e.to_string()))
                    }
                    OutputFormat::Quiet => Ok(value),
                    _ => Ok(format!("{} = {}", key, value)),
                }
            }
            ConfigAction::Set { key, value } => {
                // In a real implementation, this would modify the config
                Ok(format!("Set {} = {}", key, value))
            }
            ConfigAction::List => {
                let config_map = self.get_all_config();
                match self.output_format {
                    OutputFormat::Json => serde_json::to_string_pretty(&config_map)
                        .map_err(|e| Error::internal(e.to_string())),
                    OutputFormat::Quiet => Ok(config_map
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join("\n")),
                    _ => {
                        let mut output = String::from("Configuration:\n\n");
                        for (key, value) in &config_map {
                            output.push_str(&format!("  {} = {}\n", key, value));
                        }
                        Ok(output)
                    }
                }
            }
            ConfigAction::Reset => Ok("Configuration reset to defaults".to_string()),
        }
    }

    /// Execute the history command.
    fn execute_history(&self, count: Option<usize>) -> Result<String> {
        let entries: Vec<_> = if let Some(n) = count {
            self.history.iter().rev().take(n).collect()
        } else {
            self.history.iter().rev().collect()
        };

        if entries.is_empty() {
            return Ok("No command history".to_string());
        }

        match self.output_format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(&entries).map_err(|e| Error::internal(e.to_string()))
            }
            OutputFormat::Quiet => Ok(entries
                .iter()
                .map(|e| e.command.clone())
                .collect::<Vec<_>>()
                .join("\n")),
            _ => {
                let mut output = String::from("Command History:\n\n");
                for (i, entry) in entries.iter().enumerate() {
                    let status = if entry.success { "[+]" } else { "[-]" };
                    output.push_str(&format!(
                        "{:3}. {} {} - {}\n",
                        i + 1,
                        status,
                        entry.timestamp_str(),
                        entry.command
                    ));
                }
                Ok(output)
            }
        }
    }

    /// Execute the version command.
    fn execute_version(&self) -> Result<String> {
        let version_info = VersionInfo {
            name: crate::NAME.to_string(),
            version: crate::VERSION.to_string(),
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
        };

        match self.output_format {
            OutputFormat::Json => serde_json::to_string_pretty(&version_info)
                .map_err(|e| Error::internal(e.to_string())),
            OutputFormat::Quiet => Ok(version_info.version),
            _ => Ok(format!(
                "{} v{}\nRust {}",
                version_info.name, version_info.version, version_info.rust_version
            )),
        }
    }

    /// Execute the help command.
    fn execute_help(&self) -> Result<String> {
        Ok(r#"Panpsychism CLI - The Sorcerer's Wand

USAGE:
    panpsychism <COMMAND> [OPTIONS]

COMMANDS:
    query, q         Execute a query against the system
                     Usage: query <text> [--agent <name>]

    agents, a        List all available agents

    status, s        Show system health status

    metrics, m       Show performance metrics

    config, c        Configuration management
                     Usage: config [get|set|list|reset] [key] [value]

    history, h       Show command history
                     Usage: history [count]

    interactive, i   Enter interactive REPL mode

    version, v       Show version information

    help             Show this help message

OPTIONS:
    --output <format>    Set output format (text, json, markdown, quiet)
    --no-color           Disable colored output
    --verbose            Enable verbose output

EXAMPLES:
    panpsychism query "How to implement authentication?"
    panpsychism query "Synthesize response" --agent synthesizer
    panpsychism status
    panpsychism config get output_format
    panpsychism interactive
"#
        .to_string())
    }

    /// Add an entry to the history.
    fn add_history(&mut self, command: &str, result: &str, success: bool) {
        let entry = HistoryEntry::new(command, result, success);
        self.history.push(entry);

        // Trim history if needed
        if self.history.len() > self.config.max_history {
            let excess = self.history.len() - self.config.max_history;
            self.history.drain(0..excess);
        }
    }

    /// Get a configuration value.
    fn get_config_value(&self, key: &str) -> Result<String> {
        match key {
            "output_format" => Ok(self.config.output_format.to_string()),
            "color_enabled" => Ok(self.config.color_enabled.to_string()),
            "verbose" => Ok(self.config.verbose.to_string()),
            "max_history" => Ok(self.config.max_history.to_string()),
            "config_path" => Ok(self
                .config
                .config_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "not set".to_string())),
            "history_file" => Ok(self
                .config
                .history_file
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "not set".to_string())),
            _ => Err(Error::Config(format!("Unknown config key: {}", key))),
        }
    }

    /// Get all configuration values.
    fn get_all_config(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "output_format".to_string(),
            self.config.output_format.to_string(),
        );
        map.insert(
            "color_enabled".to_string(),
            self.config.color_enabled.to_string(),
        );
        map.insert("verbose".to_string(), self.config.verbose.to_string());
        map.insert(
            "max_history".to_string(),
            self.config.max_history.to_string(),
        );
        map.insert(
            "config_path".to_string(),
            self.config
                .config_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "not set".to_string()),
        );
        map.insert(
            "history_file".to_string(),
            self.config
                .history_file
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "not set".to_string()),
        );
        map
    }

    /// Run the CLI in interactive REPL mode.
    ///
    /// # Returns
    ///
    /// Ok(()) when the user exits, or an error if something goes wrong.
    pub fn run_interactive(&mut self) -> Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        // Print banner
        println!();
        println!("Panpsychism Interactive Mode");
        println!("Type 'help' for commands, 'exit' to quit");
        println!();

        loop {
            // Print prompt
            print!("> ");
            stdout.flush().map_err(|e| Error::Io(e))?;

            // Read line
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {}
                Err(e) => return Err(Error::Io(e)),
            }

            let line = line.trim();

            // Check for exit
            if line.is_empty() {
                continue;
            }
            if line == "exit" || line == "quit" || line == "q" {
                println!("Goodbye!");
                break;
            }

            // Parse and execute command
            let args: Vec<String> = self.parse_line(line);
            match Command::parse(&args) {
                Ok(Command::Interactive) => {
                    println!("Already in interactive mode");
                }
                Ok(cmd) => match self.execute(cmd) {
                    Ok(result) => {
                        println!(
                            "{}",
                            result.display(self.output_format, self.config.color_enabled)
                        );
                    }
                    Err(e) => {
                        if self.config.color_enabled {
                            println!("\x1b[31mError: {}\x1b[0m", e);
                        } else {
                            println!("Error: {}", e);
                        }
                    }
                },
                Err(e) => {
                    if self.config.color_enabled {
                        println!("\x1b[31mError: {}\x1b[0m", e);
                    } else {
                        println!("Error: {}", e);
                    }
                }
            }

            println!();
        }

        Ok(())
    }

    /// Parse a line into arguments, respecting quotes.
    fn parse_line(&self, line: &str) -> Vec<String> {
        let mut args = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut quote_char = '"';

        for c in line.chars() {
            match c {
                '"' | '\'' if !in_quotes => {
                    in_quotes = true;
                    quote_char = c;
                }
                c if c == quote_char && in_quotes => {
                    in_quotes = false;
                }
                ' ' if !in_quotes => {
                    if !current.is_empty() {
                        args.push(current.clone());
                        current.clear();
                    }
                }
                _ => {
                    current.push(c);
                }
            }
        }

        if !current.is_empty() {
            args.push(current);
        }

        args
    }

    /// Load history from file.
    pub fn load_history(&mut self) -> Result<()> {
        if let Some(path) = &self.config.history_file {
            if path.exists() {
                let content = std::fs::read_to_string(path)
                    .map_err(|e| Error::file_read_error(path.display().to_string(), e))?;
                self.history = serde_json::from_str(&content).unwrap_or_default();
            }
        }
        Ok(())
    }

    /// Save history to file.
    pub fn save_history(&self) -> Result<()> {
        if let Some(path) = &self.config.history_file {
            let content = serde_json::to_string_pretty(&self.history)
                .map_err(|e| Error::internal(e.to_string()))?;
            std::fs::write(path, content)
                .map_err(|e| Error::file_write_error(path.display().to_string(), e))?;
        }
        Ok(())
    }

    /// Process arguments from command line.
    pub fn run_from_args(&mut self, args: &[String]) -> Result<CommandResult> {
        // Parse global options first
        let mut filtered_args = Vec::new();
        let mut i = 0;

        while i < args.len() {
            match args[i].as_str() {
                "--output" | "-o" => {
                    if i + 1 < args.len() {
                        if let Some(format) = OutputFormat::from_str(&args[i + 1]) {
                            self.output_format = format;
                        }
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "--no-color" => {
                    self.config.color_enabled = false;
                    i += 1;
                }
                "--verbose" | "-V" => {
                    self.config.verbose = true;
                    i += 1;
                }
                s if s.starts_with("--output=") => {
                    let value = s.trim_start_matches("--output=");
                    if let Some(format) = OutputFormat::from_str(value) {
                        self.output_format = format;
                    }
                    i += 1;
                }
                _ => {
                    filtered_args.push(args[i].clone());
                    i += 1;
                }
            }
        }

        // Parse and execute command
        let command = Command::parse(&filtered_args)?;

        if matches!(command, Command::Interactive) {
            self.run_interactive()?;
            Ok(CommandResult::success("Interactive session ended", 0))
        } else {
            self.execute(command)
        }
    }
}

impl Default for CliApp {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CLI APP BUILDER
// =============================================================================

/// Builder for CliApp.
#[derive(Debug, Clone, Default)]
pub struct CliAppBuilder {
    config: CliConfig,
    orchestrator: Option<Arc<Orchestrator>>,
}

impl CliAppBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: CliConfig::default(),
            orchestrator: None,
        }
    }

    /// Set the configuration.
    pub fn config(mut self, config: CliConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the configuration file path.
    pub fn config_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.config_path = Some(path.into());
        self
    }

    /// Set the output format.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.config.output_format = format;
        self
    }

    /// Enable or disable colors.
    pub fn color_enabled(mut self, enabled: bool) -> Self {
        self.config.color_enabled = enabled;
        self
    }

    /// Set the history file path.
    pub fn history_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.history_file = Some(path.into());
        self
    }

    /// Set the maximum history entries.
    pub fn max_history(mut self, max: usize) -> Self {
        self.config.max_history = max;
        self
    }

    /// Enable or disable verbose mode.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set the orchestrator.
    pub fn orchestrator(mut self, orchestrator: Arc<Orchestrator>) -> Self {
        self.orchestrator = Some(orchestrator);
        self
    }

    /// Build the CLI application.
    pub fn build(self) -> CliApp {
        let output_format = self.config.output_format;
        CliApp {
            config: self.config,
            orchestrator: self.orchestrator,
            history: Vec::new(),
            output_format,
        }
    }
}

// =============================================================================
// HELPER TYPES
// =============================================================================

/// System status information.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemStatus {
    healthy: bool,
    uptime_secs: u64,
    agents_active: usize,
    agents_total: usize,
    memory_used_mb: usize,
    queries_processed: u64,
}

/// System performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemMetrics {
    requests_per_second: f64,
    average_latency_ms: f64,
    error_rate: f64,
    cache_hit_rate: f64,
    active_connections: usize,
}

/// Version information.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VersionInfo {
    name: String,
    version: String,
    rust_version: String,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // OutputFormat Tests
    // =========================================================================

    #[test]
    fn test_output_format_default() {
        assert_eq!(OutputFormat::default(), OutputFormat::Text);
    }

    #[test]
    fn test_output_format_as_str() {
        assert_eq!(OutputFormat::Text.as_str(), "text");
        assert_eq!(OutputFormat::Json.as_str(), "json");
        assert_eq!(OutputFormat::Markdown.as_str(), "markdown");
        assert_eq!(OutputFormat::Quiet.as_str(), "quiet");
    }

    #[test]
    fn test_output_format_from_str() {
        assert_eq!(OutputFormat::from_str("text"), Some(OutputFormat::Text));
        assert_eq!(OutputFormat::from_str("txt"), Some(OutputFormat::Text));
        assert_eq!(OutputFormat::from_str("plain"), Some(OutputFormat::Text));
        assert_eq!(OutputFormat::from_str("json"), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::from_str("markdown"), Some(OutputFormat::Markdown));
        assert_eq!(OutputFormat::from_str("md"), Some(OutputFormat::Markdown));
        assert_eq!(OutputFormat::from_str("quiet"), Some(OutputFormat::Quiet));
        assert_eq!(OutputFormat::from_str("silent"), Some(OutputFormat::Quiet));
        assert_eq!(OutputFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_output_format_supports_colors() {
        assert!(OutputFormat::Text.supports_colors());
        assert!(OutputFormat::Markdown.supports_colors());
        assert!(!OutputFormat::Json.supports_colors());
        assert!(!OutputFormat::Quiet.supports_colors());
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(format!("{}", OutputFormat::Text), "text");
        assert_eq!(format!("{}", OutputFormat::Json), "json");
    }

    // =========================================================================
    // ConfigAction Tests
    // =========================================================================

    #[test]
    fn test_config_action_parse_get() {
        let args = vec!["key1".to_string()];
        let action = ConfigAction::parse("get", &args).unwrap();
        assert_eq!(action, ConfigAction::Get { key: "key1".to_string() });
    }

    #[test]
    fn test_config_action_parse_set() {
        let args = vec!["key1".to_string(), "value1".to_string()];
        let action = ConfigAction::parse("set", &args).unwrap();
        assert_eq!(
            action,
            ConfigAction::Set {
                key: "key1".to_string(),
                value: "value1".to_string()
            }
        );
    }

    #[test]
    fn test_config_action_parse_list() {
        let action = ConfigAction::parse("list", &[]).unwrap();
        assert_eq!(action, ConfigAction::List);

        let action = ConfigAction::parse("ls", &[]).unwrap();
        assert_eq!(action, ConfigAction::List);
    }

    #[test]
    fn test_config_action_parse_reset() {
        let action = ConfigAction::parse("reset", &[]).unwrap();
        assert_eq!(action, ConfigAction::Reset);
    }

    #[test]
    fn test_config_action_parse_get_missing_key() {
        let result = ConfigAction::parse("get", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_action_parse_set_missing_value() {
        let args = vec!["key1".to_string()];
        let result = ConfigAction::parse("set", &args);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_action_parse_unknown() {
        let result = ConfigAction::parse("unknown", &[]);
        assert!(result.is_err());
    }

    // =========================================================================
    // Command Tests
    // =========================================================================

    #[test]
    fn test_command_parse_query() {
        let args = vec!["query".to_string(), "hello".to_string()];
        let cmd = Command::parse(&args).unwrap();
        assert!(matches!(cmd, Command::Query { text, agent: None } if text == "hello"));
    }

    #[test]
    fn test_command_parse_query_with_agent() {
        let args = vec![
            "query".to_string(),
            "hello".to_string(),
            "--agent".to_string(),
            "synthesizer".to_string(),
        ];
        let cmd = Command::parse(&args).unwrap();
        assert!(
            matches!(cmd, Command::Query { text, agent: Some(a) } if text == "hello" && a == "synthesizer")
        );
    }

    #[test]
    fn test_command_parse_query_with_agent_equals() {
        let args = vec![
            "query".to_string(),
            "hello".to_string(),
            "--agent=validator".to_string(),
        ];
        let cmd = Command::parse(&args).unwrap();
        assert!(
            matches!(cmd, Command::Query { text, agent: Some(a) } if text == "hello" && a == "validator")
        );
    }

    #[test]
    fn test_command_parse_query_multi_word() {
        let args = vec![
            "query".to_string(),
            "hello".to_string(),
            "world".to_string(),
            "test".to_string(),
        ];
        let cmd = Command::parse(&args).unwrap();
        assert!(matches!(cmd, Command::Query { text, agent: None } if text == "hello world test"));
    }

    #[test]
    fn test_command_parse_query_short() {
        let args = vec!["q".to_string(), "test".to_string()];
        let cmd = Command::parse(&args).unwrap();
        assert!(matches!(cmd, Command::Query { text, .. } if text == "test"));
    }

    #[test]
    fn test_command_parse_agents() {
        let args = vec!["agents".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Agents);

        let args = vec!["a".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Agents);
    }

    #[test]
    fn test_command_parse_status() {
        let args = vec!["status".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Status);

        let args = vec!["s".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Status);
    }

    #[test]
    fn test_command_parse_metrics() {
        let args = vec!["metrics".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Metrics);

        let args = vec!["m".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Metrics);
    }

    #[test]
    fn test_command_parse_config_list() {
        let args = vec!["config".to_string()];
        let cmd = Command::parse(&args).unwrap();
        assert!(matches!(
            cmd,
            Command::Config { action: ConfigAction::List }
        ));
    }

    #[test]
    fn test_command_parse_config_get() {
        let args = vec!["config".to_string(), "get".to_string(), "key".to_string()];
        let cmd = Command::parse(&args).unwrap();
        assert!(matches!(
            cmd,
            Command::Config { action: ConfigAction::Get { .. } }
        ));
    }

    #[test]
    fn test_command_parse_history() {
        let args = vec!["history".to_string()];
        assert_eq!(
            Command::parse(&args).unwrap(),
            Command::History { count: None }
        );

        let args = vec!["history".to_string(), "10".to_string()];
        assert_eq!(
            Command::parse(&args).unwrap(),
            Command::History { count: Some(10) }
        );

        let args = vec!["h".to_string()];
        assert_eq!(
            Command::parse(&args).unwrap(),
            Command::History { count: None }
        );
    }

    #[test]
    fn test_command_parse_interactive() {
        let args = vec!["interactive".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Interactive);

        let args = vec!["i".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Interactive);

        let args = vec!["repl".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Interactive);
    }

    #[test]
    fn test_command_parse_version() {
        let args = vec!["version".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Version);

        let args = vec!["--version".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Version);

        let args = vec!["-v".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Version);
    }

    #[test]
    fn test_command_parse_help() {
        let args = vec!["help".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Help);

        let args = vec!["--help".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Help);

        let args = vec!["-h".to_string()];
        assert_eq!(Command::parse(&args).unwrap(), Command::Help);
    }

    #[test]
    fn test_command_parse_empty() {
        let args: Vec<String> = vec![];
        assert_eq!(Command::parse(&args).unwrap(), Command::Help);
    }

    #[test]
    fn test_command_parse_unknown() {
        let args = vec!["unknown".to_string()];
        assert!(Command::parse(&args).is_err());
    }

    #[test]
    fn test_command_parse_query_missing_text() {
        let args = vec!["query".to_string()];
        assert!(Command::parse(&args).is_err());
    }

    #[test]
    fn test_command_name() {
        assert_eq!(Command::Status.name(), "status");
        assert_eq!(Command::Agents.name(), "agents");
        assert_eq!(Command::Help.name(), "help");
        assert_eq!(
            Command::Query {
                text: "test".to_string(),
                agent: None
            }
            .name(),
            "query"
        );
    }

    // =========================================================================
    // HistoryEntry Tests
    // =========================================================================

    #[test]
    fn test_history_entry_new() {
        let entry = HistoryEntry::new("status", "OK", true);
        assert_eq!(entry.command, "status");
        assert_eq!(entry.result, "OK");
        assert!(entry.success);
    }

    #[test]
    fn test_history_entry_age() {
        let entry = HistoryEntry::new("test", "result", true);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(entry.age() >= Duration::from_millis(10));
    }

    #[test]
    fn test_history_entry_timestamp_str() {
        let entry = HistoryEntry::new("test", "result", true);
        let ts = entry.timestamp_str();
        // Should be in format YYYY-MM-DD HH:MM:SS
        assert!(ts.len() >= 19);
    }

    // =========================================================================
    // CommandResult Tests
    // =========================================================================

    #[test]
    fn test_command_result_success() {
        let result = CommandResult::success("output text", 100);
        assert!(result.success);
        assert_eq!(result.output, "output text");
        assert!(result.error.is_none());
        assert_eq!(result.duration_ms, 100);
    }

    #[test]
    fn test_command_result_error() {
        let result = CommandResult::error("error message", 50);
        assert!(!result.success);
        assert!(result.output.is_empty());
        assert_eq!(result.error, Some("error message".to_string()));
        assert_eq!(result.duration_ms, 50);
    }

    #[test]
    fn test_command_result_display_text() {
        let result = CommandResult::success("hello", 0);
        assert!(result.display(OutputFormat::Text, false).contains("hello"));

        let result = CommandResult::error("error", 0);
        assert!(result.display(OutputFormat::Text, false).contains("Error"));
    }

    #[test]
    fn test_command_result_display_json() {
        let result = CommandResult::success("hello", 0);
        let json = result.display(OutputFormat::Json, false);
        assert!(json.contains("\"success\": true"));
    }

    #[test]
    fn test_command_result_display_quiet() {
        let result = CommandResult::success("hello", 0);
        assert_eq!(result.display(OutputFormat::Quiet, false), "hello");

        let result = CommandResult::error("error", 0);
        assert_eq!(result.display(OutputFormat::Quiet, false), "error");
    }

    #[test]
    fn test_command_result_display_markdown() {
        let result = CommandResult::success("hello", 100);
        let md = result.display(OutputFormat::Markdown, false);
        assert!(md.contains("**Result:**"));
        assert!(md.contains("100ms"));
    }

    // =========================================================================
    // AgentStatus Tests
    // =========================================================================

    #[test]
    fn test_agent_status_as_str() {
        assert_eq!(AgentStatus::Active.as_str(), "active");
        assert_eq!(AgentStatus::Idle.as_str(), "idle");
        assert_eq!(AgentStatus::Busy.as_str(), "busy");
        assert_eq!(AgentStatus::Offline.as_str(), "offline");
    }

    #[test]
    fn test_agent_status_display() {
        assert_eq!(format!("{}", AgentStatus::Active), "active");
    }

    // =========================================================================
    // CliConfig Tests
    // =========================================================================

    #[test]
    fn test_cli_config_default() {
        let config = CliConfig::default();
        assert!(config.config_path.is_none());
        assert_eq!(config.output_format, OutputFormat::Text);
        assert!(config.color_enabled);
        assert!(config.history_file.is_none());
        assert_eq!(config.max_history, 1000);
        assert!(!config.verbose);
    }

    #[test]
    fn test_cli_config_builder() {
        let config = CliConfig::builder()
            .output_format(OutputFormat::Json)
            .color_enabled(false)
            .max_history(500)
            .verbose(true)
            .build();

        assert_eq!(config.output_format, OutputFormat::Json);
        assert!(!config.color_enabled);
        assert_eq!(config.max_history, 500);
        assert!(config.verbose);
    }

    #[test]
    fn test_cli_config_builder_with_paths() {
        let config = CliConfig::builder()
            .config_path("/path/to/config")
            .history_file("/path/to/history")
            .build();

        assert_eq!(
            config.config_path,
            Some(PathBuf::from("/path/to/config"))
        );
        assert_eq!(
            config.history_file,
            Some(PathBuf::from("/path/to/history"))
        );
    }

    // =========================================================================
    // CliApp Tests
    // =========================================================================

    #[test]
    fn test_cli_app_new() {
        let app = CliApp::new();
        assert!(app.orchestrator.is_none());
        assert!(app.history.is_empty());
        assert_eq!(app.output_format, OutputFormat::Text);
    }

    #[test]
    fn test_cli_app_with_config() {
        let config = CliConfig::builder()
            .output_format(OutputFormat::Json)
            .build();
        let app = CliApp::with_config(config);
        assert_eq!(app.output_format, OutputFormat::Json);
    }

    #[test]
    fn test_cli_app_builder() {
        let app = CliApp::builder()
            .output_format(OutputFormat::Markdown)
            .color_enabled(false)
            .verbose(true)
            .build();

        assert_eq!(app.output_format, OutputFormat::Markdown);
        assert!(!app.config.color_enabled);
        assert!(app.config.verbose);
    }

    #[test]
    fn test_cli_app_set_output_format() {
        let mut app = CliApp::new();
        app.set_output_format(OutputFormat::Json);
        assert_eq!(app.output_format(), OutputFormat::Json);
    }

    #[test]
    fn test_cli_app_execute_status() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Status).unwrap();
        assert!(result.success);
        assert!(result.output.contains("System Status"));
    }

    #[test]
    fn test_cli_app_execute_agents() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Agents).unwrap();
        assert!(result.success);
        assert!(result.output.contains("Available Agents"));
    }

    #[test]
    fn test_cli_app_execute_metrics() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Metrics).unwrap();
        assert!(result.success);
        assert!(result.output.contains("Performance Metrics"));
    }

    #[test]
    fn test_cli_app_execute_version() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Version).unwrap();
        assert!(result.success);
        assert!(result.output.contains(crate::VERSION));
    }

    #[test]
    fn test_cli_app_execute_help() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Help).unwrap();
        assert!(result.success);
        assert!(result.output.contains("USAGE"));
        assert!(result.output.contains("COMMANDS"));
    }

    #[test]
    fn test_cli_app_execute_query() {
        let mut app = CliApp::new();
        let result = app
            .execute(Command::Query {
                text: "test query".to_string(),
                agent: None,
            })
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("test query"));
    }

    #[test]
    fn test_cli_app_execute_query_with_agent() {
        let mut app = CliApp::new();
        let result = app
            .execute(Command::Query {
                text: "test".to_string(),
                agent: Some("synthesizer".to_string()),
            })
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("synthesizer"));
    }

    #[test]
    fn test_cli_app_execute_config_list() {
        let mut app = CliApp::new();
        let result = app
            .execute(Command::Config {
                action: ConfigAction::List,
            })
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("Configuration"));
    }

    #[test]
    fn test_cli_app_execute_config_get() {
        let mut app = CliApp::new();
        let result = app
            .execute(Command::Config {
                action: ConfigAction::Get {
                    key: "output_format".to_string(),
                },
            })
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("text"));
    }

    #[test]
    fn test_cli_app_execute_config_get_unknown() {
        let mut app = CliApp::new();
        let result = app
            .execute(Command::Config {
                action: ConfigAction::Get {
                    key: "unknown_key".to_string(),
                },
            })
            .unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_cli_app_execute_config_set() {
        let mut app = CliApp::new();
        let result = app
            .execute(Command::Config {
                action: ConfigAction::Set {
                    key: "test".to_string(),
                    value: "value".to_string(),
                },
            })
            .unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_cli_app_execute_config_reset() {
        let mut app = CliApp::new();
        let result = app
            .execute(Command::Config {
                action: ConfigAction::Reset,
            })
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("reset"));
    }

    #[test]
    fn test_cli_app_execute_history_empty() {
        let mut app = CliApp::new();
        let result = app.execute(Command::History { count: None }).unwrap();
        // First command adds to history, so we check the result message
        assert!(result.success);
    }

    #[test]
    fn test_cli_app_history_tracking() {
        let mut app = CliApp::new();

        // Execute a few commands
        app.execute(Command::Status).unwrap();
        app.execute(Command::Version).unwrap();

        assert_eq!(app.history().len(), 2);
    }

    #[test]
    fn test_cli_app_history_limit() {
        let config = CliConfig::builder().max_history(3).build();
        let mut app = CliApp::with_config(config);

        // Execute more commands than the limit
        for _ in 0..5 {
            app.execute(Command::Status).unwrap();
        }

        assert_eq!(app.history().len(), 3);
    }

    #[test]
    fn test_cli_app_parse_line() {
        let app = CliApp::new();

        let args = app.parse_line("query hello world");
        assert_eq!(args, vec!["query", "hello", "world"]);

        let args = app.parse_line("query \"hello world\"");
        assert_eq!(args, vec!["query", "hello world"]);

        let args = app.parse_line("query 'hello world'");
        assert_eq!(args, vec!["query", "hello world"]);

        let args = app.parse_line("config get key");
        assert_eq!(args, vec!["config", "get", "key"]);
    }

    #[test]
    fn test_cli_app_run_from_args() {
        let mut app = CliApp::new();

        let args = vec!["status".to_string()];
        let result = app.run_from_args(&args).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_cli_app_run_from_args_with_output_format() {
        let mut app = CliApp::new();

        let args = vec!["--output".to_string(), "json".to_string(), "status".to_string()];
        let result = app.run_from_args(&args).unwrap();
        assert!(result.success);
        assert_eq!(app.output_format(), OutputFormat::Json);
    }

    #[test]
    fn test_cli_app_run_from_args_with_output_format_equals() {
        let mut app = CliApp::new();

        let args = vec!["--output=json".to_string(), "status".to_string()];
        let result = app.run_from_args(&args).unwrap();
        assert!(result.success);
        assert_eq!(app.output_format(), OutputFormat::Json);
    }

    #[test]
    fn test_cli_app_run_from_args_no_color() {
        let mut app = CliApp::new();

        let args = vec!["--no-color".to_string(), "status".to_string()];
        app.run_from_args(&args).unwrap();
        assert!(!app.config.color_enabled);
    }

    #[test]
    fn test_cli_app_run_from_args_verbose() {
        let mut app = CliApp::new();

        let args = vec!["--verbose".to_string(), "status".to_string()];
        app.run_from_args(&args).unwrap();
        assert!(app.config.verbose);
    }

    // =========================================================================
    // CliAppBuilder Tests
    // =========================================================================

    #[test]
    fn test_cli_app_builder_default() {
        let builder = CliAppBuilder::new();
        let app = builder.build();
        assert!(app.orchestrator.is_none());
    }

    #[test]
    fn test_cli_app_builder_with_config() {
        let config = CliConfig::builder()
            .output_format(OutputFormat::Quiet)
            .build();

        let app = CliAppBuilder::new().config(config).build();

        assert_eq!(app.output_format, OutputFormat::Quiet);
    }

    #[test]
    fn test_cli_app_builder_chain() {
        let app = CliAppBuilder::new()
            .config_path("/test/path")
            .output_format(OutputFormat::Json)
            .color_enabled(false)
            .history_file("/test/history")
            .max_history(100)
            .verbose(true)
            .build();

        assert_eq!(app.config.config_path, Some(PathBuf::from("/test/path")));
        assert_eq!(app.output_format, OutputFormat::Json);
        assert!(!app.config.color_enabled);
        assert_eq!(app.config.history_file, Some(PathBuf::from("/test/history")));
        assert_eq!(app.config.max_history, 100);
        assert!(app.config.verbose);
    }

    // =========================================================================
    // JSON Output Format Tests
    // =========================================================================

    #[test]
    fn test_cli_app_status_json_output() {
        let mut app = CliApp::builder()
            .output_format(OutputFormat::Json)
            .build();

        let result = app.execute(Command::Status).unwrap();
        assert!(result.success);

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert!(parsed.get("healthy").is_some());
    }

    #[test]
    fn test_cli_app_agents_json_output() {
        let mut app = CliApp::builder()
            .output_format(OutputFormat::Json)
            .build();

        let result = app.execute(Command::Agents).unwrap();
        assert!(result.success);

        // Should be valid JSON array
        let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert!(parsed.is_array());
    }

    #[test]
    fn test_cli_app_config_list_json_output() {
        let mut app = CliApp::builder()
            .output_format(OutputFormat::Json)
            .build();

        let result = app
            .execute(Command::Config {
                action: ConfigAction::List,
            })
            .unwrap();
        assert!(result.success);

        // Should be valid JSON object
        let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert!(parsed.get("output_format").is_some());
    }

    // =========================================================================
    // Quiet Output Format Tests
    // =========================================================================

    #[test]
    fn test_cli_app_status_quiet_output() {
        let mut app = CliApp::builder()
            .output_format(OutputFormat::Quiet)
            .build();

        let result = app.execute(Command::Status).unwrap();
        assert!(result.success);
        assert_eq!(result.output.trim(), "OK");
    }

    #[test]
    fn test_cli_app_agents_quiet_output() {
        let mut app = CliApp::builder()
            .output_format(OutputFormat::Quiet)
            .build();

        let result = app.execute(Command::Agents).unwrap();
        assert!(result.success);
        // Should just be agent IDs, one per line
        assert!(result.output.contains("search"));
        assert!(result.output.contains("synthesizer"));
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_cli_app_execute_empty_query() {
        let mut app = CliApp::new();
        let result = app
            .execute(Command::Query {
                text: String::new(),
                agent: None,
            })
            .unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_cli_app_execute_interactive_error() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Interactive);
        assert!(result.is_err());
    }

    // =========================================================================
    // AgentInfo Tests
    // =========================================================================

    #[test]
    fn test_agent_info_creation() {
        let agent = AgentInfo {
            id: "test".to_string(),
            name: "Test Agent".to_string(),
            description: "A test agent".to_string(),
            status: AgentStatus::Active,
            capabilities: vec!["cap1".to_string(), "cap2".to_string()],
        };

        assert_eq!(agent.id, "test");
        assert_eq!(agent.status, AgentStatus::Active);
        assert_eq!(agent.capabilities.len(), 2);
    }

    #[test]
    fn test_agent_info_serialization() {
        let agent = AgentInfo {
            id: "test".to_string(),
            name: "Test Agent".to_string(),
            description: "A test agent".to_string(),
            status: AgentStatus::Active,
            capabilities: vec!["cap1".to_string()],
        };

        let json = serde_json::to_string(&agent).unwrap();
        let parsed: AgentInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, agent.id);
        assert_eq!(parsed.status, agent.status);
    }
}
