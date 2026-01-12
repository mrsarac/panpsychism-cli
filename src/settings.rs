//! # Settings Module
//!
//! Unified configuration management with TOML support, environment variables, and runtime updates.
//!
//! ## Overview
//!
//! The settings module provides a comprehensive configuration system for Panpsychism with:
//!
//! - **TOML file support**: Load and save configuration from/to TOML files
//! - **Environment variables**: Override settings via `PANPSYCHISM_` prefixed env vars
//! - **Runtime updates**: Modify settings dynamically during execution
//! - **Validation**: Ensure configuration consistency and correctness
//! - **Builder pattern**: Fluent API for constructing settings
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use panpsychism::settings::{Settings, Environment};
//!
//! // Load with sensible defaults
//! let settings = Settings::default();
//!
//! // Load from file with env overlay
//! let mut settings = Settings::load_from_file("config.toml")?;
//! settings.merge_with_env();
//!
//! // Use builder for custom configuration
//! let settings = Settings::builder()
//!     .environment(Environment::Production)
//!     .debug(false)
//!     .agent_timeout_ms(60000)
//!     .build();
//! ```
//!
//! ## Environment Variables
//!
//! All settings can be overridden via environment variables with the `PANPSYCHISM_` prefix:
//!
//! | Variable | Setting |
//! |----------|---------|
//! | `PANPSYCHISM_DEBUG` | `general.debug` |
//! | `PANPSYCHISM_ENVIRONMENT` | `general.environment` |
//! | `PANPSYCHISM_LOG_LEVEL` | `logging.level` |
//! | `PANPSYCHISM_AGENT_TIMEOUT_MS` | `agents.default_timeout_ms` |
//!
//! ## Configuration File Example
//!
//! ```toml
//! [general]
//! name = "panpsychism"
//! environment = "development"
//! debug = true
//!
//! [agents]
//! default_timeout_ms = 30000
//! max_concurrent = 10
//!
//! [bus]
//! max_queue_size = 1000
//! heartbeat_interval_ms = 5000
//!
//! [memory]
//! max_short_term = 10000
//! persistence_path = "./data/memory.json"
//!
//! [logging]
//! level = "info"
//! format = "pretty"
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during settings operations.
#[derive(Debug, Error)]
pub enum SettingsError {
    /// Failed to read configuration file.
    #[error("Failed to read configuration file: {0}")]
    FileRead(#[from] std::io::Error),

    /// Failed to parse TOML content.
    #[error("Failed to parse TOML: {0}")]
    TomlParse(#[from] toml::de::Error),

    /// Failed to serialize to TOML.
    #[error("Failed to serialize to TOML: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    /// Validation error in settings.
    #[error("Settings validation failed: {0}")]
    Validation(String),

    /// Invalid setting key.
    #[error("Invalid setting key: {0}")]
    InvalidKey(String),

    /// Type conversion error.
    #[error("Type conversion error for key '{key}': {message}")]
    TypeConversion { key: String, message: String },
}

/// Result type for settings operations.
pub type SettingsResult<T> = Result<T, SettingsError>;

// ============================================================================
// Environment Enum
// ============================================================================

/// Runtime environment classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    /// Development environment with verbose logging and debug features.
    #[default]
    Development,
    /// Staging environment for pre-production testing.
    Staging,
    /// Production environment with optimized settings.
    Production,
}

impl Environment {
    /// Returns the string representation of the environment.
    pub fn as_str(&self) -> &'static str {
        match self {
            Environment::Development => "development",
            Environment::Staging => "staging",
            Environment::Production => "production",
        }
    }

    /// Parse environment from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "development" | "dev" => Some(Environment::Development),
            "staging" | "stage" => Some(Environment::Staging),
            "production" | "prod" => Some(Environment::Production),
            _ => None,
        }
    }

    /// Check if this is a development environment.
    pub fn is_development(&self) -> bool {
        matches!(self, Environment::Development)
    }

    /// Check if this is a production environment.
    pub fn is_production(&self) -> bool {
        matches!(self, Environment::Production)
    }
}

impl std::fmt::Display for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Log Level Enum
// ============================================================================

/// Logging verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Most verbose, includes all trace information.
    Trace,
    /// Debug information for development.
    Debug,
    /// Standard informational messages.
    #[default]
    Info,
    /// Warning messages for potential issues.
    Warn,
    /// Error messages only.
    Error,
}

impl LogLevel {
    /// Returns the string representation of the log level.
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }

    /// Parse log level from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "trace" => Some(LogLevel::Trace),
            "debug" => Some(LogLevel::Debug),
            "info" => Some(LogLevel::Info),
            "warn" | "warning" => Some(LogLevel::Warn),
            "error" => Some(LogLevel::Error),
            _ => None,
        }
    }

    /// Convert to tracing level filter string.
    pub fn to_tracing_filter(&self) -> &'static str {
        self.as_str()
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Log Format Enum
// ============================================================================

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    /// Plain text format.
    Plain,
    /// JSON structured format.
    Json,
    /// Pretty-printed format with colors.
    #[default]
    Pretty,
}

impl LogFormat {
    /// Returns the string representation of the log format.
    pub fn as_str(&self) -> &'static str {
        match self {
            LogFormat::Plain => "plain",
            LogFormat::Json => "json",
            LogFormat::Pretty => "pretty",
        }
    }

    /// Parse log format from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "plain" | "text" => Some(LogFormat::Plain),
            "json" => Some(LogFormat::Json),
            "pretty" => Some(LogFormat::Pretty),
            _ => None,
        }
    }

    /// Check if format supports colors.
    pub fn supports_colors(&self) -> bool {
        matches!(self, LogFormat::Pretty)
    }
}

impl std::fmt::Display for LogFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Settings Structures
// ============================================================================

/// General application settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralSettings {
    /// Application name.
    pub name: String,
    /// Application version.
    pub version: String,
    /// Runtime environment.
    pub environment: Environment,
    /// Enable debug mode.
    pub debug: bool,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            name: "panpsychism".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            environment: Environment::Development,
            debug: false,
        }
    }
}

/// Agent-related settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSettings {
    /// Default timeout for agent operations in milliseconds.
    pub default_timeout_ms: u64,
    /// Maximum number of concurrent agent operations.
    pub max_concurrent: usize,
    /// Interval for health checks in milliseconds.
    pub health_check_interval_ms: u64,
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
            default_timeout_ms: 30_000,
            max_concurrent: 10,
            health_check_interval_ms: 5_000,
        }
    }
}

/// Message bus settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusSettings {
    /// Maximum number of messages in the queue.
    pub max_queue_size: usize,
    /// Maximum number of log entries to retain.
    pub max_log_size: usize,
    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
}

impl Default for BusSettings {
    fn default() -> Self {
        Self {
            max_queue_size: 1_000,
            max_log_size: 10_000,
            heartbeat_interval_ms: 5_000,
        }
    }
}

/// Memory management settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySettings {
    /// Maximum entries in short-term memory.
    pub max_short_term: usize,
    /// Maximum entries in long-term memory.
    pub max_long_term: usize,
    /// Path for memory persistence (optional).
    pub persistence_path: Option<String>,
    /// Auto-save interval in seconds.
    pub auto_save_seconds: u64,
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            max_short_term: 10_000,
            max_long_term: 100_000,
            persistence_path: None,
            auto_save_seconds: 300, // 5 minutes
        }
    }
}

/// Logging configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// Log verbosity level.
    pub level: LogLevel,
    /// Log output format.
    pub format: LogFormat,
    /// Optional file path for log output.
    pub file_path: Option<String>,
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Pretty,
            file_path: None,
        }
    }
}

// ============================================================================
// Main Settings Structure
// ============================================================================

/// Unified configuration settings for Panpsychism.
///
/// This structure holds all configuration options organized into logical sections:
///
/// - `general`: Application-wide settings (name, version, environment)
/// - `agents`: Agent operation settings (timeouts, concurrency)
/// - `bus`: Message bus settings (queue sizes, heartbeat)
/// - `memory`: Memory management settings (limits, persistence)
/// - `logging`: Logging configuration (level, format, output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// General application settings.
    pub general: GeneralSettings,
    /// Agent-related settings.
    pub agents: AgentSettings,
    /// Message bus settings.
    pub bus: BusSettings,
    /// Memory management settings.
    pub memory: MemorySettings,
    /// Logging configuration.
    pub logging: LoggingSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            general: GeneralSettings::default(),
            agents: AgentSettings::default(),
            bus: BusSettings::default(),
            memory: MemorySettings::default(),
            logging: LoggingSettings::default(),
        }
    }
}

impl Settings {
    /// Create a new SettingsBuilder for fluent configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::{Settings, Environment};
    ///
    /// let settings = Settings::builder()
    ///     .environment(Environment::Production)
    ///     .debug(false)
    ///     .log_level(panpsychism::settings::LogLevel::Warn)
    ///     .build();
    /// ```
    pub fn builder() -> SettingsBuilder {
        SettingsBuilder::new()
    }

    /// Load settings from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the TOML configuration file
    ///
    /// # Returns
    ///
    /// * `Ok(Settings)` - Successfully loaded settings
    /// * `Err(SettingsError)` - If file cannot be read or parsed
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let settings = Settings::load_from_file("config.toml")?;
    /// ```
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> SettingsResult<Self> {
        let content = fs::read_to_string(path)?;
        Self::load_from_toml(&content)
    }

    /// Load settings from a TOML string.
    ///
    /// # Arguments
    ///
    /// * `content` - TOML content as a string
    ///
    /// # Returns
    ///
    /// * `Ok(Settings)` - Successfully parsed settings
    /// * `Err(SettingsError)` - If TOML parsing fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::Settings;
    ///
    /// let toml = r#"
    /// [general]
    /// debug = true
    ///
    /// [agents]
    /// default_timeout_ms = 60000
    /// "#;
    ///
    /// let settings = Settings::load_from_toml(toml).unwrap();
    /// assert!(settings.general.debug);
    /// ```
    pub fn load_from_toml(content: &str) -> SettingsResult<Self> {
        // Parse with defaults for missing fields
        let parsed: PartialSettings = toml::from_str(content)?;
        Ok(parsed.into_settings())
    }

    /// Load settings from environment variables.
    ///
    /// All environment variables use the `PANPSYCHISM_` prefix.
    /// Missing variables use default values.
    ///
    /// # Supported Variables
    ///
    /// | Variable | Type | Default |
    /// |----------|------|---------|
    /// | `PANPSYCHISM_NAME` | String | "panpsychism" |
    /// | `PANPSYCHISM_ENVIRONMENT` | String | "development" |
    /// | `PANPSYCHISM_DEBUG` | bool | false |
    /// | `PANPSYCHISM_AGENT_TIMEOUT_MS` | u64 | 30000 |
    /// | `PANPSYCHISM_MAX_CONCURRENT` | usize | 10 |
    /// | `PANPSYCHISM_LOG_LEVEL` | String | "info" |
    /// | `PANPSYCHISM_LOG_FORMAT` | String | "pretty" |
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::Settings;
    ///
    /// // Set environment variable
    /// std::env::set_var("PANPSYCHISM_DEBUG", "true");
    ///
    /// let settings = Settings::load_from_env();
    /// // debug will be true if env var was set
    ///
    /// // Cleanup
    /// std::env::remove_var("PANPSYCHISM_DEBUG");
    /// ```
    pub fn load_from_env() -> Self {
        let mut settings = Self::default();
        settings.merge_with_env();
        settings
    }

    /// Merge current settings with environment variable overrides.
    ///
    /// Environment variables take precedence over current values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::Settings;
    ///
    /// let mut settings = Settings::default();
    /// std::env::set_var("PANPSYCHISM_DEBUG", "true");
    /// settings.merge_with_env();
    /// assert!(settings.general.debug);
    /// std::env::remove_var("PANPSYCHISM_DEBUG");
    /// ```
    pub fn merge_with_env(&mut self) {
        // General settings
        if let Ok(name) = env::var("PANPSYCHISM_NAME") {
            self.general.name = name;
        }
        if let Ok(version) = env::var("PANPSYCHISM_VERSION") {
            self.general.version = version;
        }
        if let Ok(env_str) = env::var("PANPSYCHISM_ENVIRONMENT") {
            if let Some(environment) = Environment::from_str(&env_str) {
                self.general.environment = environment;
            }
        }
        if let Ok(debug) = env::var("PANPSYCHISM_DEBUG") {
            self.general.debug = debug.parse().unwrap_or(self.general.debug);
        }

        // Agent settings
        if let Ok(timeout) = env::var("PANPSYCHISM_AGENT_TIMEOUT_MS") {
            if let Ok(v) = timeout.parse() {
                self.agents.default_timeout_ms = v;
            }
        }
        if let Ok(concurrent) = env::var("PANPSYCHISM_MAX_CONCURRENT") {
            if let Ok(v) = concurrent.parse() {
                self.agents.max_concurrent = v;
            }
        }
        if let Ok(health) = env::var("PANPSYCHISM_HEALTH_CHECK_INTERVAL_MS") {
            if let Ok(v) = health.parse() {
                self.agents.health_check_interval_ms = v;
            }
        }

        // Bus settings
        if let Ok(queue) = env::var("PANPSYCHISM_MAX_QUEUE_SIZE") {
            if let Ok(v) = queue.parse() {
                self.bus.max_queue_size = v;
            }
        }
        if let Ok(log_size) = env::var("PANPSYCHISM_MAX_LOG_SIZE") {
            if let Ok(v) = log_size.parse() {
                self.bus.max_log_size = v;
            }
        }
        if let Ok(heartbeat) = env::var("PANPSYCHISM_HEARTBEAT_INTERVAL_MS") {
            if let Ok(v) = heartbeat.parse() {
                self.bus.heartbeat_interval_ms = v;
            }
        }

        // Memory settings
        if let Ok(short) = env::var("PANPSYCHISM_MAX_SHORT_TERM") {
            if let Ok(v) = short.parse() {
                self.memory.max_short_term = v;
            }
        }
        if let Ok(long) = env::var("PANPSYCHISM_MAX_LONG_TERM") {
            if let Ok(v) = long.parse() {
                self.memory.max_long_term = v;
            }
        }
        if let Ok(path) = env::var("PANPSYCHISM_PERSISTENCE_PATH") {
            self.memory.persistence_path = Some(path);
        }
        if let Ok(auto_save) = env::var("PANPSYCHISM_AUTO_SAVE_SECONDS") {
            if let Ok(v) = auto_save.parse() {
                self.memory.auto_save_seconds = v;
            }
        }

        // Logging settings
        if let Ok(level) = env::var("PANPSYCHISM_LOG_LEVEL") {
            if let Some(l) = LogLevel::from_str(&level) {
                self.logging.level = l;
            }
        }
        if let Ok(format) = env::var("PANPSYCHISM_LOG_FORMAT") {
            if let Some(f) = LogFormat::from_str(&format) {
                self.logging.format = f;
            }
        }
        if let Ok(path) = env::var("PANPSYCHISM_LOG_FILE") {
            self.logging.file_path = Some(path);
        }
    }

    /// Save settings to a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the configuration file should be saved
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Settings saved successfully
    /// * `Err(SettingsError)` - If file cannot be written
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let settings = Settings::default();
    /// settings.save_to_file("config.toml")?;
    /// ```
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> SettingsResult<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Get a setting value by dot-notation key.
    ///
    /// # Arguments
    ///
    /// * `key` - Dot-notation key (e.g., "general.debug", "agents.default_timeout_ms")
    ///
    /// # Returns
    ///
    /// * `Some(String)` - String representation of the value
    /// * `None` - If key is invalid
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::Settings;
    ///
    /// let settings = Settings::default();
    /// assert_eq!(settings.get("general.name"), Some("panpsychism".to_string()));
    /// assert_eq!(settings.get("invalid.key"), None);
    /// ```
    pub fn get(&self, key: &str) -> Option<String> {
        match key {
            // General
            "general.name" => Some(self.general.name.clone()),
            "general.version" => Some(self.general.version.clone()),
            "general.environment" => Some(self.general.environment.to_string()),
            "general.debug" => Some(self.general.debug.to_string()),

            // Agents
            "agents.default_timeout_ms" => Some(self.agents.default_timeout_ms.to_string()),
            "agents.max_concurrent" => Some(self.agents.max_concurrent.to_string()),
            "agents.health_check_interval_ms" => {
                Some(self.agents.health_check_interval_ms.to_string())
            }

            // Bus
            "bus.max_queue_size" => Some(self.bus.max_queue_size.to_string()),
            "bus.max_log_size" => Some(self.bus.max_log_size.to_string()),
            "bus.heartbeat_interval_ms" => Some(self.bus.heartbeat_interval_ms.to_string()),

            // Memory
            "memory.max_short_term" => Some(self.memory.max_short_term.to_string()),
            "memory.max_long_term" => Some(self.memory.max_long_term.to_string()),
            "memory.persistence_path" => self.memory.persistence_path.clone(),
            "memory.auto_save_seconds" => Some(self.memory.auto_save_seconds.to_string()),

            // Logging
            "logging.level" => Some(self.logging.level.to_string()),
            "logging.format" => Some(self.logging.format.to_string()),
            "logging.file_path" => self.logging.file_path.clone(),

            _ => None,
        }
    }

    /// Set a setting value by dot-notation key.
    ///
    /// # Arguments
    ///
    /// * `key` - Dot-notation key (e.g., "general.debug", "agents.default_timeout_ms")
    /// * `value` - String value to set
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Value set successfully
    /// * `Err(SettingsError)` - If key is invalid or value cannot be parsed
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::Settings;
    ///
    /// let mut settings = Settings::default();
    /// settings.set("general.debug", "true").unwrap();
    /// assert!(settings.general.debug);
    /// ```
    pub fn set(&mut self, key: &str, value: &str) -> SettingsResult<()> {
        match key {
            // General
            "general.name" => {
                self.general.name = value.to_string();
                Ok(())
            }
            "general.version" => {
                self.general.version = value.to_string();
                Ok(())
            }
            "general.environment" => {
                self.general.environment = Environment::from_str(value).ok_or_else(|| {
                    SettingsError::TypeConversion {
                        key: key.to_string(),
                        message: format!("Invalid environment: {}", value),
                    }
                })?;
                Ok(())
            }
            "general.debug" => {
                self.general.debug =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid bool: {}", value),
                        })?;
                Ok(())
            }

            // Agents
            "agents.default_timeout_ms" => {
                self.agents.default_timeout_ms =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid u64: {}", value),
                        })?;
                Ok(())
            }
            "agents.max_concurrent" => {
                self.agents.max_concurrent =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid usize: {}", value),
                        })?;
                Ok(())
            }
            "agents.health_check_interval_ms" => {
                self.agents.health_check_interval_ms =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid u64: {}", value),
                        })?;
                Ok(())
            }

            // Bus
            "bus.max_queue_size" => {
                self.bus.max_queue_size =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid usize: {}", value),
                        })?;
                Ok(())
            }
            "bus.max_log_size" => {
                self.bus.max_log_size =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid usize: {}", value),
                        })?;
                Ok(())
            }
            "bus.heartbeat_interval_ms" => {
                self.bus.heartbeat_interval_ms =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid u64: {}", value),
                        })?;
                Ok(())
            }

            // Memory
            "memory.max_short_term" => {
                self.memory.max_short_term =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid usize: {}", value),
                        })?;
                Ok(())
            }
            "memory.max_long_term" => {
                self.memory.max_long_term =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid usize: {}", value),
                        })?;
                Ok(())
            }
            "memory.persistence_path" => {
                self.memory.persistence_path = if value.is_empty() {
                    None
                } else {
                    Some(value.to_string())
                };
                Ok(())
            }
            "memory.auto_save_seconds" => {
                self.memory.auto_save_seconds =
                    value
                        .parse()
                        .map_err(|_| SettingsError::TypeConversion {
                            key: key.to_string(),
                            message: format!("Invalid u64: {}", value),
                        })?;
                Ok(())
            }

            // Logging
            "logging.level" => {
                self.logging.level =
                    LogLevel::from_str(value).ok_or_else(|| SettingsError::TypeConversion {
                        key: key.to_string(),
                        message: format!("Invalid log level: {}", value),
                    })?;
                Ok(())
            }
            "logging.format" => {
                self.logging.format =
                    LogFormat::from_str(value).ok_or_else(|| SettingsError::TypeConversion {
                        key: key.to_string(),
                        message: format!("Invalid log format: {}", value),
                    })?;
                Ok(())
            }
            "logging.file_path" => {
                self.logging.file_path = if value.is_empty() {
                    None
                } else {
                    Some(value.to_string())
                };
                Ok(())
            }

            _ => Err(SettingsError::InvalidKey(key.to_string())),
        }
    }

    /// Validate all settings for consistency and correctness.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - All settings are valid
    /// * `Err(SettingsError)` - If validation fails
    ///
    /// # Validation Rules
    ///
    /// - Name must not be empty
    /// - Version must not be empty
    /// - Timeout must be > 0
    /// - Max concurrent must be > 0
    /// - Queue sizes must be > 0
    /// - Memory limits must be > 0
    /// - Auto-save interval must be > 0
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::Settings;
    ///
    /// let settings = Settings::default();
    /// assert!(settings.validate().is_ok());
    /// ```
    pub fn validate(&self) -> SettingsResult<()> {
        let mut errors = Vec::new();

        // General validation
        if self.general.name.is_empty() {
            errors.push("general.name cannot be empty");
        }
        if self.general.version.is_empty() {
            errors.push("general.version cannot be empty");
        }

        // Agent validation
        if self.agents.default_timeout_ms == 0 {
            errors.push("agents.default_timeout_ms must be > 0");
        }
        if self.agents.max_concurrent == 0 {
            errors.push("agents.max_concurrent must be > 0");
        }
        if self.agents.health_check_interval_ms == 0 {
            errors.push("agents.health_check_interval_ms must be > 0");
        }

        // Bus validation
        if self.bus.max_queue_size == 0 {
            errors.push("bus.max_queue_size must be > 0");
        }
        if self.bus.max_log_size == 0 {
            errors.push("bus.max_log_size must be > 0");
        }
        if self.bus.heartbeat_interval_ms == 0 {
            errors.push("bus.heartbeat_interval_ms must be > 0");
        }

        // Memory validation
        if self.memory.max_short_term == 0 {
            errors.push("memory.max_short_term must be > 0");
        }
        if self.memory.max_long_term == 0 {
            errors.push("memory.max_long_term must be > 0");
        }
        if self.memory.auto_save_seconds == 0 {
            errors.push("memory.auto_save_seconds must be > 0");
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(SettingsError::Validation(errors.join("; ")))
        }
    }

    /// Get all settings as a flat key-value map.
    ///
    /// # Returns
    ///
    /// A HashMap with dot-notation keys and string values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::Settings;
    ///
    /// let settings = Settings::default();
    /// let map = settings.to_map();
    /// assert!(map.contains_key("general.name"));
    /// ```
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();

        // General
        map.insert("general.name".to_string(), self.general.name.clone());
        map.insert("general.version".to_string(), self.general.version.clone());
        map.insert(
            "general.environment".to_string(),
            self.general.environment.to_string(),
        );
        map.insert("general.debug".to_string(), self.general.debug.to_string());

        // Agents
        map.insert(
            "agents.default_timeout_ms".to_string(),
            self.agents.default_timeout_ms.to_string(),
        );
        map.insert(
            "agents.max_concurrent".to_string(),
            self.agents.max_concurrent.to_string(),
        );
        map.insert(
            "agents.health_check_interval_ms".to_string(),
            self.agents.health_check_interval_ms.to_string(),
        );

        // Bus
        map.insert(
            "bus.max_queue_size".to_string(),
            self.bus.max_queue_size.to_string(),
        );
        map.insert(
            "bus.max_log_size".to_string(),
            self.bus.max_log_size.to_string(),
        );
        map.insert(
            "bus.heartbeat_interval_ms".to_string(),
            self.bus.heartbeat_interval_ms.to_string(),
        );

        // Memory
        map.insert(
            "memory.max_short_term".to_string(),
            self.memory.max_short_term.to_string(),
        );
        map.insert(
            "memory.max_long_term".to_string(),
            self.memory.max_long_term.to_string(),
        );
        if let Some(ref path) = self.memory.persistence_path {
            map.insert("memory.persistence_path".to_string(), path.clone());
        }
        map.insert(
            "memory.auto_save_seconds".to_string(),
            self.memory.auto_save_seconds.to_string(),
        );

        // Logging
        map.insert(
            "logging.level".to_string(),
            self.logging.level.to_string(),
        );
        map.insert(
            "logging.format".to_string(),
            self.logging.format.to_string(),
        );
        if let Some(ref path) = self.logging.file_path {
            map.insert("logging.file_path".to_string(), path.clone());
        }

        map
    }

    /// Create production-optimized settings.
    ///
    /// Returns settings tuned for production use:
    /// - No debug mode
    /// - Production environment
    /// - Warn-level logging
    /// - Larger queue sizes
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::{Settings, Environment};
    ///
    /// let settings = Settings::production();
    /// assert_eq!(settings.general.environment, Environment::Production);
    /// assert!(!settings.general.debug);
    /// ```
    pub fn production() -> Self {
        Self {
            general: GeneralSettings {
                environment: Environment::Production,
                debug: false,
                ..Default::default()
            },
            agents: AgentSettings {
                default_timeout_ms: 60_000,
                max_concurrent: 50,
                health_check_interval_ms: 10_000,
            },
            bus: BusSettings {
                max_queue_size: 10_000,
                max_log_size: 100_000,
                heartbeat_interval_ms: 10_000,
            },
            memory: MemorySettings {
                max_short_term: 50_000,
                max_long_term: 500_000,
                persistence_path: Some("./data/memory.json".to_string()),
                auto_save_seconds: 60,
            },
            logging: LoggingSettings {
                level: LogLevel::Warn,
                format: LogFormat::Json,
                file_path: Some("./logs/panpsychism.log".to_string()),
            },
        }
    }

    /// Create development-optimized settings.
    ///
    /// Returns settings tuned for development:
    /// - Debug mode enabled
    /// - Development environment
    /// - Debug-level logging with pretty format
    ///
    /// # Example
    ///
    /// ```rust
    /// use panpsychism::settings::{Settings, Environment};
    ///
    /// let settings = Settings::development();
    /// assert_eq!(settings.general.environment, Environment::Development);
    /// assert!(settings.general.debug);
    /// ```
    pub fn development() -> Self {
        Self {
            general: GeneralSettings {
                environment: Environment::Development,
                debug: true,
                ..Default::default()
            },
            logging: LoggingSettings {
                level: LogLevel::Debug,
                format: LogFormat::Pretty,
                file_path: None,
            },
            ..Default::default()
        }
    }
}

// ============================================================================
// Partial Settings for TOML Parsing
// ============================================================================

/// Partial settings structure for TOML parsing with optional fields.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PartialSettings {
    #[serde(default)]
    general: PartialGeneralSettings,
    #[serde(default)]
    agents: PartialAgentSettings,
    #[serde(default)]
    bus: PartialBusSettings,
    #[serde(default)]
    memory: PartialMemorySettings,
    #[serde(default)]
    logging: PartialLoggingSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PartialGeneralSettings {
    name: Option<String>,
    version: Option<String>,
    environment: Option<Environment>,
    debug: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PartialAgentSettings {
    default_timeout_ms: Option<u64>,
    max_concurrent: Option<usize>,
    health_check_interval_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PartialBusSettings {
    max_queue_size: Option<usize>,
    max_log_size: Option<usize>,
    heartbeat_interval_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PartialMemorySettings {
    max_short_term: Option<usize>,
    max_long_term: Option<usize>,
    persistence_path: Option<String>,
    auto_save_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PartialLoggingSettings {
    level: Option<LogLevel>,
    format: Option<LogFormat>,
    file_path: Option<String>,
}

impl PartialSettings {
    fn into_settings(self) -> Settings {
        let defaults = Settings::default();

        Settings {
            general: GeneralSettings {
                name: self.general.name.unwrap_or(defaults.general.name),
                version: self.general.version.unwrap_or(defaults.general.version),
                environment: self
                    .general
                    .environment
                    .unwrap_or(defaults.general.environment),
                debug: self.general.debug.unwrap_or(defaults.general.debug),
            },
            agents: AgentSettings {
                default_timeout_ms: self
                    .agents
                    .default_timeout_ms
                    .unwrap_or(defaults.agents.default_timeout_ms),
                max_concurrent: self
                    .agents
                    .max_concurrent
                    .unwrap_or(defaults.agents.max_concurrent),
                health_check_interval_ms: self
                    .agents
                    .health_check_interval_ms
                    .unwrap_or(defaults.agents.health_check_interval_ms),
            },
            bus: BusSettings {
                max_queue_size: self
                    .bus
                    .max_queue_size
                    .unwrap_or(defaults.bus.max_queue_size),
                max_log_size: self.bus.max_log_size.unwrap_or(defaults.bus.max_log_size),
                heartbeat_interval_ms: self
                    .bus
                    .heartbeat_interval_ms
                    .unwrap_or(defaults.bus.heartbeat_interval_ms),
            },
            memory: MemorySettings {
                max_short_term: self
                    .memory
                    .max_short_term
                    .unwrap_or(defaults.memory.max_short_term),
                max_long_term: self
                    .memory
                    .max_long_term
                    .unwrap_or(defaults.memory.max_long_term),
                persistence_path: self.memory.persistence_path.or(defaults.memory.persistence_path),
                auto_save_seconds: self
                    .memory
                    .auto_save_seconds
                    .unwrap_or(defaults.memory.auto_save_seconds),
            },
            logging: LoggingSettings {
                level: self.logging.level.unwrap_or(defaults.logging.level),
                format: self.logging.format.unwrap_or(defaults.logging.format),
                file_path: self.logging.file_path.or(defaults.logging.file_path),
            },
        }
    }
}

// ============================================================================
// Settings Builder
// ============================================================================

/// Fluent builder for constructing Settings.
///
/// # Example
///
/// ```rust
/// use panpsychism::settings::{Settings, SettingsBuilder, Environment, LogLevel};
///
/// let settings = SettingsBuilder::new()
///     .name("my-app")
///     .environment(Environment::Production)
///     .debug(false)
///     .agent_timeout_ms(60000)
///     .max_concurrent(20)
///     .log_level(LogLevel::Warn)
///     .build();
///
/// assert_eq!(settings.general.name, "my-app");
/// assert_eq!(settings.agents.default_timeout_ms, 60000);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SettingsBuilder {
    settings: Settings,
}

impl SettingsBuilder {
    /// Create a new SettingsBuilder with default values.
    pub fn new() -> Self {
        Self {
            settings: Settings::default(),
        }
    }

    /// Set the application name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.settings.general.name = name.into();
        self
    }

    /// Set the application version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.settings.general.version = version.into();
        self
    }

    /// Set the runtime environment.
    pub fn environment(mut self, environment: Environment) -> Self {
        self.settings.general.environment = environment;
        self
    }

    /// Enable or disable debug mode.
    pub fn debug(mut self, debug: bool) -> Self {
        self.settings.general.debug = debug;
        self
    }

    /// Set the default agent timeout in milliseconds.
    pub fn agent_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.settings.agents.default_timeout_ms = timeout_ms;
        self
    }

    /// Set the maximum concurrent agent operations.
    pub fn max_concurrent(mut self, max: usize) -> Self {
        self.settings.agents.max_concurrent = max;
        self
    }

    /// Set the health check interval in milliseconds.
    pub fn health_check_interval_ms(mut self, interval_ms: u64) -> Self {
        self.settings.agents.health_check_interval_ms = interval_ms;
        self
    }

    /// Set the maximum message queue size.
    pub fn max_queue_size(mut self, size: usize) -> Self {
        self.settings.bus.max_queue_size = size;
        self
    }

    /// Set the maximum log size.
    pub fn max_log_size(mut self, size: usize) -> Self {
        self.settings.bus.max_log_size = size;
        self
    }

    /// Set the heartbeat interval in milliseconds.
    pub fn heartbeat_interval_ms(mut self, interval_ms: u64) -> Self {
        self.settings.bus.heartbeat_interval_ms = interval_ms;
        self
    }

    /// Set the maximum short-term memory entries.
    pub fn max_short_term(mut self, max: usize) -> Self {
        self.settings.memory.max_short_term = max;
        self
    }

    /// Set the maximum long-term memory entries.
    pub fn max_long_term(mut self, max: usize) -> Self {
        self.settings.memory.max_long_term = max;
        self
    }

    /// Set the persistence path for memory.
    pub fn persistence_path(mut self, path: impl Into<String>) -> Self {
        self.settings.memory.persistence_path = Some(path.into());
        self
    }

    /// Set the auto-save interval in seconds.
    pub fn auto_save_seconds(mut self, seconds: u64) -> Self {
        self.settings.memory.auto_save_seconds = seconds;
        self
    }

    /// Set the log level.
    pub fn log_level(mut self, level: LogLevel) -> Self {
        self.settings.logging.level = level;
        self
    }

    /// Set the log format.
    pub fn log_format(mut self, format: LogFormat) -> Self {
        self.settings.logging.format = format;
        self
    }

    /// Set the log file path.
    pub fn log_file(mut self, path: impl Into<String>) -> Self {
        self.settings.logging.file_path = Some(path.into());
        self
    }

    /// Build the Settings instance.
    pub fn build(self) -> Settings {
        self.settings
    }

    /// Build and validate the Settings instance.
    ///
    /// # Returns
    ///
    /// * `Ok(Settings)` - Valid settings
    /// * `Err(SettingsError)` - If validation fails
    pub fn build_validated(self) -> SettingsResult<Settings> {
        let settings = self.settings;
        settings.validate()?;
        Ok(settings)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    // ========================================================================
    // Environment Enum Tests
    // ========================================================================

    #[test]
    fn test_environment_default() {
        assert_eq!(Environment::default(), Environment::Development);
    }

    #[test]
    fn test_environment_as_str() {
        assert_eq!(Environment::Development.as_str(), "development");
        assert_eq!(Environment::Staging.as_str(), "staging");
        assert_eq!(Environment::Production.as_str(), "production");
    }

    #[test]
    fn test_environment_from_str() {
        assert_eq!(
            Environment::from_str("development"),
            Some(Environment::Development)
        );
        assert_eq!(Environment::from_str("dev"), Some(Environment::Development));
        assert_eq!(Environment::from_str("staging"), Some(Environment::Staging));
        assert_eq!(Environment::from_str("stage"), Some(Environment::Staging));
        assert_eq!(
            Environment::from_str("production"),
            Some(Environment::Production)
        );
        assert_eq!(Environment::from_str("prod"), Some(Environment::Production));
        assert_eq!(Environment::from_str("invalid"), None);
    }

    #[test]
    fn test_environment_is_methods() {
        assert!(Environment::Development.is_development());
        assert!(!Environment::Development.is_production());
        assert!(Environment::Production.is_production());
        assert!(!Environment::Production.is_development());
    }

    #[test]
    fn test_environment_display() {
        assert_eq!(format!("{}", Environment::Development), "development");
        assert_eq!(format!("{}", Environment::Production), "production");
    }

    // ========================================================================
    // LogLevel Enum Tests
    // ========================================================================

    #[test]
    fn test_log_level_default() {
        assert_eq!(LogLevel::default(), LogLevel::Info);
    }

    #[test]
    fn test_log_level_as_str() {
        assert_eq!(LogLevel::Trace.as_str(), "trace");
        assert_eq!(LogLevel::Debug.as_str(), "debug");
        assert_eq!(LogLevel::Info.as_str(), "info");
        assert_eq!(LogLevel::Warn.as_str(), "warn");
        assert_eq!(LogLevel::Error.as_str(), "error");
    }

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(LogLevel::from_str("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::from_str("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_str("info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("warn"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("warning"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("invalid"), None);
    }

    #[test]
    fn test_log_level_to_tracing_filter() {
        assert_eq!(LogLevel::Debug.to_tracing_filter(), "debug");
    }

    // ========================================================================
    // LogFormat Enum Tests
    // ========================================================================

    #[test]
    fn test_log_format_default() {
        assert_eq!(LogFormat::default(), LogFormat::Pretty);
    }

    #[test]
    fn test_log_format_as_str() {
        assert_eq!(LogFormat::Plain.as_str(), "plain");
        assert_eq!(LogFormat::Json.as_str(), "json");
        assert_eq!(LogFormat::Pretty.as_str(), "pretty");
    }

    #[test]
    fn test_log_format_from_str() {
        assert_eq!(LogFormat::from_str("plain"), Some(LogFormat::Plain));
        assert_eq!(LogFormat::from_str("text"), Some(LogFormat::Plain));
        assert_eq!(LogFormat::from_str("json"), Some(LogFormat::Json));
        assert_eq!(LogFormat::from_str("pretty"), Some(LogFormat::Pretty));
        assert_eq!(LogFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_log_format_supports_colors() {
        assert!(!LogFormat::Plain.supports_colors());
        assert!(!LogFormat::Json.supports_colors());
        assert!(LogFormat::Pretty.supports_colors());
    }

    // ========================================================================
    // Settings Default Tests
    // ========================================================================

    #[test]
    fn test_settings_default() {
        let settings = Settings::default();
        assert_eq!(settings.general.name, "panpsychism");
        assert!(!settings.general.debug);
        assert_eq!(settings.general.environment, Environment::Development);
        assert_eq!(settings.agents.default_timeout_ms, 30_000);
        assert_eq!(settings.agents.max_concurrent, 10);
        assert_eq!(settings.bus.max_queue_size, 1_000);
        assert_eq!(settings.memory.max_short_term, 10_000);
        assert_eq!(settings.logging.level, LogLevel::Info);
    }

    #[test]
    fn test_settings_production() {
        let settings = Settings::production();
        assert_eq!(settings.general.environment, Environment::Production);
        assert!(!settings.general.debug);
        assert_eq!(settings.agents.default_timeout_ms, 60_000);
        assert_eq!(settings.agents.max_concurrent, 50);
        assert_eq!(settings.logging.level, LogLevel::Warn);
        assert_eq!(settings.logging.format, LogFormat::Json);
    }

    #[test]
    fn test_settings_development() {
        let settings = Settings::development();
        assert_eq!(settings.general.environment, Environment::Development);
        assert!(settings.general.debug);
        assert_eq!(settings.logging.level, LogLevel::Debug);
        assert_eq!(settings.logging.format, LogFormat::Pretty);
    }

    // ========================================================================
    // TOML Loading Tests
    // ========================================================================

    #[test]
    fn test_load_from_toml_complete() {
        let toml = r#"
[general]
name = "test-app"
version = "2.0.0"
environment = "production"
debug = true

[agents]
default_timeout_ms = 60000
max_concurrent = 20
health_check_interval_ms = 10000

[bus]
max_queue_size = 5000
max_log_size = 50000
heartbeat_interval_ms = 3000

[memory]
max_short_term = 20000
max_long_term = 200000
persistence_path = "/data/memory.json"
auto_save_seconds = 120

[logging]
level = "warn"
format = "json"
file_path = "/var/log/app.log"
"#;

        let settings = Settings::load_from_toml(toml).unwrap();
        assert_eq!(settings.general.name, "test-app");
        assert_eq!(settings.general.version, "2.0.0");
        assert_eq!(settings.general.environment, Environment::Production);
        assert!(settings.general.debug);
        assert_eq!(settings.agents.default_timeout_ms, 60000);
        assert_eq!(settings.agents.max_concurrent, 20);
        assert_eq!(settings.bus.max_queue_size, 5000);
        assert_eq!(settings.memory.max_short_term, 20000);
        assert_eq!(
            settings.memory.persistence_path,
            Some("/data/memory.json".to_string())
        );
        assert_eq!(settings.logging.level, LogLevel::Warn);
        assert_eq!(settings.logging.format, LogFormat::Json);
    }

    #[test]
    fn test_load_from_toml_partial() {
        let toml = r#"
[general]
debug = true

[agents]
default_timeout_ms = 45000
"#;

        let settings = Settings::load_from_toml(toml).unwrap();
        assert!(settings.general.debug);
        assert_eq!(settings.general.name, "panpsychism"); // default
        assert_eq!(settings.agents.default_timeout_ms, 45000);
        assert_eq!(settings.agents.max_concurrent, 10); // default
    }

    #[test]
    fn test_load_from_toml_empty() {
        let toml = "";
        let settings = Settings::load_from_toml(toml).unwrap();
        assert_eq!(settings.general.name, "panpsychism");
        assert!(!settings.general.debug);
    }

    #[test]
    fn test_load_from_toml_invalid() {
        let toml = "this is not valid toml [[[";
        let result = Settings::load_from_toml(toml);
        assert!(result.is_err());
    }

    // ========================================================================
    // Environment Variable Tests
    // ========================================================================

    #[test]
    fn test_merge_with_env_debug() {
        let mut settings = Settings::default();
        env::set_var("PANPSYCHISM_DEBUG", "true");
        settings.merge_with_env();
        assert!(settings.general.debug);
        env::remove_var("PANPSYCHISM_DEBUG");
    }

    #[test]
    fn test_merge_with_env_environment() {
        let mut settings = Settings::default();
        env::set_var("PANPSYCHISM_ENVIRONMENT", "production");
        settings.merge_with_env();
        assert_eq!(settings.general.environment, Environment::Production);
        env::remove_var("PANPSYCHISM_ENVIRONMENT");
    }

    #[test]
    fn test_merge_with_env_timeout() {
        let mut settings = Settings::default();
        env::set_var("PANPSYCHISM_AGENT_TIMEOUT_MS", "99000");
        settings.merge_with_env();
        assert_eq!(settings.agents.default_timeout_ms, 99000);
        env::remove_var("PANPSYCHISM_AGENT_TIMEOUT_MS");
    }

    #[test]
    fn test_merge_with_env_log_level() {
        let mut settings = Settings::default();
        env::set_var("PANPSYCHISM_LOG_LEVEL", "error");
        settings.merge_with_env();
        assert_eq!(settings.logging.level, LogLevel::Error);
        env::remove_var("PANPSYCHISM_LOG_LEVEL");
    }

    #[test]
    fn test_merge_with_env_invalid_value() {
        let mut settings = Settings::default();
        env::set_var("PANPSYCHISM_DEBUG", "not_a_bool");
        settings.merge_with_env();
        assert!(!settings.general.debug); // keeps default
        env::remove_var("PANPSYCHISM_DEBUG");
    }

    #[test]
    fn test_load_from_env() {
        env::set_var("PANPSYCHISM_NAME", "env-app");
        env::set_var("PANPSYCHISM_DEBUG", "true");
        let settings = Settings::load_from_env();
        assert_eq!(settings.general.name, "env-app");
        assert!(settings.general.debug);
        env::remove_var("PANPSYCHISM_NAME");
        env::remove_var("PANPSYCHISM_DEBUG");
    }

    // ========================================================================
    // Get/Set Tests
    // ========================================================================

    #[test]
    fn test_get_valid_keys() {
        let settings = Settings::default();
        assert_eq!(settings.get("general.name"), Some("panpsychism".to_string()));
        assert_eq!(settings.get("general.debug"), Some("false".to_string()));
        assert_eq!(
            settings.get("agents.default_timeout_ms"),
            Some("30000".to_string())
        );
        assert_eq!(settings.get("logging.level"), Some("info".to_string()));
    }

    #[test]
    fn test_get_invalid_key() {
        let settings = Settings::default();
        assert_eq!(settings.get("invalid.key"), None);
        assert_eq!(settings.get("general.nonexistent"), None);
    }

    #[test]
    fn test_set_valid_keys() {
        let mut settings = Settings::default();

        settings.set("general.name", "new-name").unwrap();
        assert_eq!(settings.general.name, "new-name");

        settings.set("general.debug", "true").unwrap();
        assert!(settings.general.debug);

        settings.set("agents.default_timeout_ms", "50000").unwrap();
        assert_eq!(settings.agents.default_timeout_ms, 50000);

        settings.set("logging.level", "error").unwrap();
        assert_eq!(settings.logging.level, LogLevel::Error);
    }

    #[test]
    fn test_set_invalid_key() {
        let mut settings = Settings::default();
        let result = settings.set("invalid.key", "value");
        assert!(matches!(result, Err(SettingsError::InvalidKey(_))));
    }

    #[test]
    fn test_set_invalid_value() {
        let mut settings = Settings::default();
        let result = settings.set("general.debug", "not_bool");
        assert!(matches!(result, Err(SettingsError::TypeConversion { .. })));
    }

    #[test]
    fn test_set_environment() {
        let mut settings = Settings::default();
        settings.set("general.environment", "production").unwrap();
        assert_eq!(settings.general.environment, Environment::Production);
    }

    #[test]
    fn test_set_optional_paths() {
        let mut settings = Settings::default();

        settings
            .set("memory.persistence_path", "/data/mem.json")
            .unwrap();
        assert_eq!(
            settings.memory.persistence_path,
            Some("/data/mem.json".to_string())
        );

        settings.set("memory.persistence_path", "").unwrap();
        assert_eq!(settings.memory.persistence_path, None);
    }

    // ========================================================================
    // Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_default_passes() {
        let settings = Settings::default();
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_name_fails() {
        let mut settings = Settings::default();
        settings.general.name = "".to_string();
        let result = settings.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("name cannot be empty"));
    }

    #[test]
    fn test_validate_zero_timeout_fails() {
        let mut settings = Settings::default();
        settings.agents.default_timeout_ms = 0;
        let result = settings.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("default_timeout_ms must be > 0"));
    }

    #[test]
    fn test_validate_zero_concurrent_fails() {
        let mut settings = Settings::default();
        settings.agents.max_concurrent = 0;
        let result = settings.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_multiple_errors() {
        let mut settings = Settings::default();
        settings.general.name = "".to_string();
        settings.agents.default_timeout_ms = 0;
        settings.bus.max_queue_size = 0;
        let result = settings.validate();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("name"));
        assert!(err.contains("timeout"));
        assert!(err.contains("queue"));
    }

    // ========================================================================
    // Builder Tests
    // ========================================================================

    #[test]
    fn test_builder_basic() {
        let settings = SettingsBuilder::new().build();
        assert_eq!(settings.general.name, "panpsychism");
    }

    #[test]
    fn test_builder_all_fields() {
        let settings = SettingsBuilder::new()
            .name("custom-app")
            .version("3.0.0")
            .environment(Environment::Staging)
            .debug(true)
            .agent_timeout_ms(45000)
            .max_concurrent(25)
            .health_check_interval_ms(8000)
            .max_queue_size(2000)
            .max_log_size(20000)
            .heartbeat_interval_ms(7000)
            .max_short_term(15000)
            .max_long_term(150000)
            .persistence_path("/custom/path.json")
            .auto_save_seconds(180)
            .log_level(LogLevel::Debug)
            .log_format(LogFormat::Json)
            .log_file("/var/log/custom.log")
            .build();

        assert_eq!(settings.general.name, "custom-app");
        assert_eq!(settings.general.version, "3.0.0");
        assert_eq!(settings.general.environment, Environment::Staging);
        assert!(settings.general.debug);
        assert_eq!(settings.agents.default_timeout_ms, 45000);
        assert_eq!(settings.agents.max_concurrent, 25);
        assert_eq!(settings.agents.health_check_interval_ms, 8000);
        assert_eq!(settings.bus.max_queue_size, 2000);
        assert_eq!(settings.bus.max_log_size, 20000);
        assert_eq!(settings.bus.heartbeat_interval_ms, 7000);
        assert_eq!(settings.memory.max_short_term, 15000);
        assert_eq!(settings.memory.max_long_term, 150000);
        assert_eq!(
            settings.memory.persistence_path,
            Some("/custom/path.json".to_string())
        );
        assert_eq!(settings.memory.auto_save_seconds, 180);
        assert_eq!(settings.logging.level, LogLevel::Debug);
        assert_eq!(settings.logging.format, LogFormat::Json);
        assert_eq!(
            settings.logging.file_path,
            Some("/var/log/custom.log".to_string())
        );
    }

    #[test]
    fn test_builder_validated_success() {
        let result = SettingsBuilder::new()
            .name("valid-app")
            .agent_timeout_ms(1000)
            .build_validated();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_validated_failure() {
        let result = SettingsBuilder::new()
            .name("") // invalid
            .build_validated();
        assert!(result.is_err());
    }

    #[test]
    fn test_settings_builder_method() {
        let settings = Settings::builder()
            .debug(true)
            .log_level(LogLevel::Trace)
            .build();
        assert!(settings.general.debug);
        assert_eq!(settings.logging.level, LogLevel::Trace);
    }

    // ========================================================================
    // To Map Tests
    // ========================================================================

    #[test]
    fn test_to_map_contains_all_keys() {
        let settings = Settings::default();
        let map = settings.to_map();

        assert!(map.contains_key("general.name"));
        assert!(map.contains_key("general.version"));
        assert!(map.contains_key("general.environment"));
        assert!(map.contains_key("general.debug"));
        assert!(map.contains_key("agents.default_timeout_ms"));
        assert!(map.contains_key("agents.max_concurrent"));
        assert!(map.contains_key("bus.max_queue_size"));
        assert!(map.contains_key("memory.max_short_term"));
        assert!(map.contains_key("logging.level"));
        assert!(map.contains_key("logging.format"));
    }

    #[test]
    fn test_to_map_optional_fields() {
        let settings = Settings::default();
        let map = settings.to_map();

        // Optional fields not present when None
        assert!(!map.contains_key("memory.persistence_path"));
        assert!(!map.contains_key("logging.file_path"));
    }

    #[test]
    fn test_to_map_with_optional_fields() {
        let settings = Settings::builder()
            .persistence_path("/data/mem.json")
            .log_file("/var/log/app.log")
            .build();
        let map = settings.to_map();

        assert_eq!(map.get("memory.persistence_path").unwrap(), "/data/mem.json");
        assert_eq!(map.get("logging.file_path").unwrap(), "/var/log/app.log");
    }

    // ========================================================================
    // File I/O Tests
    // ========================================================================

    #[test]
    fn test_save_and_load_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_config.toml");

        let settings = Settings::builder()
            .name("file-test")
            .debug(true)
            .agent_timeout_ms(55000)
            .log_level(LogLevel::Warn)
            .build();

        settings.save_to_file(&file_path).unwrap();
        let loaded = Settings::load_from_file(&file_path).unwrap();

        assert_eq!(loaded.general.name, "file-test");
        assert!(loaded.general.debug);
        assert_eq!(loaded.agents.default_timeout_ms, 55000);
        assert_eq!(loaded.logging.level, LogLevel::Warn);
    }

    #[test]
    fn test_load_file_not_found() {
        let result = Settings::load_from_file("/nonexistent/path/config.toml");
        assert!(result.is_err());
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_settings_serialize_deserialize() {
        let settings = Settings::builder()
            .name("serde-test")
            .environment(Environment::Staging)
            .debug(true)
            .build();

        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: Settings = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.general.name, "serde-test");
        assert_eq!(deserialized.general.environment, Environment::Staging);
        assert!(deserialized.general.debug);
    }

    #[test]
    fn test_environment_serde() {
        let json = "\"production\"";
        let env: Environment = serde_json::from_str(json).unwrap();
        assert_eq!(env, Environment::Production);

        let serialized = serde_json::to_string(&Environment::Staging).unwrap();
        assert_eq!(serialized, "\"staging\"");
    }

    #[test]
    fn test_log_level_serde() {
        let json = "\"debug\"";
        let level: LogLevel = serde_json::from_str(json).unwrap();
        assert_eq!(level, LogLevel::Debug);
    }

    #[test]
    fn test_log_format_serde() {
        let json = "\"json\"";
        let format: LogFormat = serde_json::from_str(json).unwrap();
        assert_eq!(format, LogFormat::Json);
    }

    // ========================================================================
    // Clone Tests
    // ========================================================================

    #[test]
    fn test_settings_clone() {
        let settings = Settings::builder().name("original").debug(true).build();

        let cloned = settings.clone();
        assert_eq!(cloned.general.name, "original");
        assert!(cloned.general.debug);
    }

    #[test]
    fn test_builder_clone() {
        let builder = SettingsBuilder::new().name("builder-clone").debug(true);

        let cloned = builder.clone();
        let settings = cloned.build();
        assert_eq!(settings.general.name, "builder-clone");
    }

    // ========================================================================
    // Error Display Tests
    // ========================================================================

    #[test]
    fn test_error_display() {
        let err = SettingsError::InvalidKey("test.key".to_string());
        assert!(err.to_string().contains("test.key"));

        let err = SettingsError::Validation("multiple errors".to_string());
        assert!(err.to_string().contains("multiple errors"));

        let err = SettingsError::TypeConversion {
            key: "general.debug".to_string(),
            message: "not a bool".to_string(),
        };
        assert!(err.to_string().contains("general.debug"));
        assert!(err.to_string().contains("not a bool"));
    }
}
