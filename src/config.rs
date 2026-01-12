//! Configuration management for Project Panpsychism.
//!
//! This module provides the [`Config`] struct that holds all configuration
//! settings for the Panpsychism library. Configuration can be loaded from
//! a YAML file or use sensible defaults.
//!
//! # Default Configuration Location
//!
//! The default configuration file is located at:
//! - Unix: `~/.config/panpsychism/config.yaml`
//! - Windows: `%APPDATA%/panpsychism/config.yaml`
//!
//! # Example Configuration File
//!
//! ```yaml
//! prompts_dir: ./prompts
//! data_dir: ./data
//! index_file: ./data/masters.mv2
//! llm_endpoint: http://127.0.0.1:8045/v1/chat/completions
//! llm_api_key: sk-antigravity
//! llm_model: gemini-3-flash
//! privacy:
//!   default_tier: internal
//!   redact_patterns: []
//!
//! # Analyze command configuration
//! analyze:
//!   default_input: stdin
//!   default_outputs:
//!     - clipboard
//!   verbose: false
//!
//! # Output routing configuration
//! output:
//!   clipboard: true
//!   file_path: null
//!   second_core: false
//!   second_core_endpoint: "https://webhook.mustafasarac.com"
//!
//! # Content type to prompt mapping
//! content_mapping:
//!   meeting: "prompts/master/meeting-notes.yaml"
//!   code: "prompts/master/code-request.yaml"
//!   blog: "prompts/master/blog-writing.yaml"
//!   note: "prompts/master/daily-note.yaml"
//!   household: "prompts/master/household.yaml"
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use panpsychism::Config;
//!
//! // Load from default location or use defaults
//! let config = Config::load()?;
//!
//! // Load from specific file
//! let config = Config::load_from_file(Path::new("./my-config.yaml"))?;
//!
//! // Save configuration
//! config.save(Path::new("./config.yaml"))?;
//! ```

// Standard library
use std::path::{Path, PathBuf};

// External crates
use serde::{Deserialize, Serialize};

// Internal modules
use crate::error::Result;
use crate::privacy::PrivacyConfig;

// =============================================================================
// ANALYZE CONFIG STRUCTS
// =============================================================================

/// Input source for analyze command.
///
/// Determines where the analyze command reads its input from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InputSource {
    /// Read from standard input (default)
    Stdin,
    /// Read from system clipboard
    Clipboard,
    /// Read from a specific file path
    File(String),
}

impl Default for InputSource {
    fn default() -> Self {
        Self::Stdin
    }
}

/// Output destination for analyze command.
///
/// Determines where the analyze command writes its output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputDestination {
    /// Write to standard output
    Stdout,
    /// Write to system clipboard
    Clipboard,
    /// Write to a file (path will be determined by OutputConfig)
    File,
    /// Send to Second Core webhook
    SecondCore,
}

impl Default for OutputDestination {
    fn default() -> Self {
        Self::Clipboard
    }
}

/// Configuration for the analyze command.
///
/// Controls input/output behavior and verbosity of the analyze command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeConfig {
    /// Default input source for content to analyze
    #[serde(default)]
    pub default_input: InputSource,

    /// Default output destinations (can have multiple)
    #[serde(default = "default_analyze_outputs")]
    pub default_outputs: Vec<OutputDestination>,

    /// Enable verbose logging during analysis
    #[serde(default)]
    pub verbose: bool,
}

fn default_analyze_outputs() -> Vec<OutputDestination> {
    vec![OutputDestination::Clipboard]
}

impl Default for AnalyzeConfig {
    fn default() -> Self {
        Self {
            default_input: InputSource::default(),
            default_outputs: default_analyze_outputs(),
            verbose: false,
        }
    }
}

// =============================================================================
// OUTPUT CONFIG STRUCT
// =============================================================================

/// Output routing configuration.
///
/// Controls how and where processed content is delivered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Whether to copy output to system clipboard
    #[serde(default = "default_true")]
    pub clipboard: bool,

    /// Optional file path for output (if None, file output disabled)
    #[serde(default)]
    pub file_path: Option<String>,

    /// Whether to send output to Second Core webhook
    #[serde(default)]
    pub second_core: bool,

    /// Second Core webhook endpoint URL
    #[serde(default = "default_second_core_endpoint")]
    pub second_core_endpoint: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_second_core_endpoint() -> Option<String> {
    Some("https://webhook.mustafasarac.com".to_string())
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            clipboard: true,
            file_path: None,
            second_core: false,
            second_core_endpoint: default_second_core_endpoint(),
        }
    }
}

// =============================================================================
// CONTENT TYPE MAPPING STRUCT
// =============================================================================

/// Maps content types to their corresponding prompt files.
///
/// Each field contains a path (relative to prompts_dir) to the prompt template
/// for that content type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTypeMapping {
    /// Prompt file for meeting notes processing
    #[serde(default = "default_meeting_prompt")]
    pub meeting: String,

    /// Prompt file for code request processing
    #[serde(default = "default_code_prompt")]
    pub code: String,

    /// Prompt file for blog writing processing
    #[serde(default = "default_blog_prompt")]
    pub blog: String,

    /// Prompt file for daily note processing
    #[serde(default = "default_note_prompt")]
    pub note: String,

    /// Prompt file for household/personal processing
    #[serde(default = "default_household_prompt")]
    pub household: String,
}

fn default_meeting_prompt() -> String {
    "prompts/master/meeting-notes.yaml".to_string()
}

fn default_code_prompt() -> String {
    "prompts/master/code-request.yaml".to_string()
}

fn default_blog_prompt() -> String {
    "prompts/master/blog-writing.yaml".to_string()
}

fn default_note_prompt() -> String {
    "prompts/master/daily-note.yaml".to_string()
}

fn default_household_prompt() -> String {
    "prompts/master/household.yaml".to_string()
}

impl Default for ContentTypeMapping {
    fn default() -> Self {
        Self {
            meeting: default_meeting_prompt(),
            code: default_code_prompt(),
            blog: default_blog_prompt(),
            note: default_note_prompt(),
            household: default_household_prompt(),
        }
    }
}

// =============================================================================
// CONFIG STRUCT
// =============================================================================

/// Main configuration struct for Project Panpsychism.
///
/// This struct holds all the configuration settings needed to run the
/// Panpsychism library, including directory paths, LLM endpoint settings,
/// and privacy configuration.
///
/// # Fields
///
/// - `prompts_dir`: Directory containing prompt templates (default: `./prompts`)
/// - `data_dir`: Directory for storing data files (default: `./data`)
/// - `index_file`: Path to the Memvid index file (default: `./data/masters.mv2`)
/// - `llm_endpoint`: LLM API endpoint URL (default: Antigravity proxy)
/// - `llm_api_key`: API key for LLM authentication
/// - `llm_model`: Model identifier to use for LLM requests
/// - `privacy`: Privacy configuration settings
/// - `analyze`: Analyze command configuration (optional)
/// - `output`: Output routing configuration (optional)
/// - `content_mapping`: Content type to prompt mapping (optional)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Directory containing prompt templates.
    ///
    /// This directory should contain markdown or text files with prompt templates
    /// organized by category.
    #[serde(default = "default_prompts_dir")]
    pub prompts_dir: PathBuf,

    /// Directory for storing data files.
    ///
    /// This includes the Memvid index and any cached data.
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Path to the Memvid index file (.mv2 format).
    ///
    /// This file contains the semantic index of all prompts for fast retrieval.
    #[serde(default = "default_index_file")]
    pub index_file: PathBuf,

    /// LLM API endpoint URL.
    ///
    /// Default points to the Antigravity proxy for local Gemini access.
    #[serde(default = "default_llm_endpoint")]
    pub llm_endpoint: String,

    /// API key for LLM authentication.
    ///
    /// Default is the Antigravity proxy key.
    #[serde(default = "default_llm_api_key")]
    pub llm_api_key: String,

    /// Model identifier for LLM requests.
    ///
    /// Default is `gemini-3-flash` via Antigravity proxy.
    #[serde(default = "default_llm_model")]
    pub llm_model: String,

    /// Privacy configuration settings.
    #[serde(default)]
    pub privacy: PrivacyConfig,

    /// Analyze command configuration.
    ///
    /// Controls input/output behavior for the analyze command.
    /// If not specified, uses default AnalyzeConfig.
    #[serde(default)]
    pub analyze: Option<AnalyzeConfig>,

    /// Output routing configuration.
    ///
    /// Controls where processed content is delivered.
    /// If not specified, uses default OutputConfig.
    #[serde(default)]
    pub output: Option<OutputConfig>,

    /// Content type to prompt file mapping.
    ///
    /// Maps content types (meeting, code, blog, etc.) to their
    /// corresponding prompt template files.
    /// If not specified, uses default ContentTypeMapping.
    #[serde(default)]
    pub content_mapping: Option<ContentTypeMapping>,
}

// =============================================================================
// DEFAULT VALUE FUNCTIONS
// =============================================================================

fn default_prompts_dir() -> PathBuf {
    PathBuf::from("./prompts")
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("./data")
}

fn default_index_file() -> PathBuf {
    PathBuf::from("./data/masters.mv2")
}

fn default_llm_endpoint() -> String {
    "http://127.0.0.1:8045/v1/chat/completions".to_string()
}

fn default_llm_api_key() -> String {
    "sk-antigravity".to_string()
}

fn default_llm_model() -> String {
    "gemini-3-flash".to_string()
}

// =============================================================================
// TRAIT IMPLEMENTATIONS
// =============================================================================

impl Default for Config {
    /// Creates a new Config with default values.
    ///
    /// # Default Values
    ///
    /// | Field | Default Value |
    /// |-------|---------------|
    /// | `prompts_dir` | `./prompts` |
    /// | `data_dir` | `./data` |
    /// | `index_file` | `./data/masters.mv2` |
    /// | `llm_endpoint` | `http://127.0.0.1:8045/v1/chat/completions` |
    /// | `llm_api_key` | `sk-antigravity` |
    /// | `llm_model` | `gemini-3-flash` |
    /// | `privacy` | Default privacy config |
    /// | `analyze` | None (uses AnalyzeConfig::default when accessed) |
    /// | `output` | None (uses OutputConfig::default when accessed) |
    /// | `content_mapping` | None (uses ContentTypeMapping::default when accessed) |
    fn default() -> Self {
        Self {
            prompts_dir: default_prompts_dir(),
            data_dir: default_data_dir(),
            index_file: default_index_file(),
            llm_endpoint: default_llm_endpoint(),
            llm_api_key: default_llm_api_key(),
            llm_model: default_llm_model(),
            privacy: PrivacyConfig::default(),
            analyze: None,
            output: None,
            content_mapping: None,
        }
    }
}

// =============================================================================
// CONFIG METHODS
// =============================================================================

impl Config {
    /// Loads configuration from the default config file or returns defaults.
    ///
    /// This method attempts to load configuration from [`Config::config_path()`].
    /// If the file doesn't exist, it returns the default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the config file exists but cannot be parsed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = Config::load()?;
    /// println!("Prompts directory: {:?}", config.prompts_dir);
    /// ```
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if config_path.exists() {
            Self::load_from_file(&config_path)
        } else {
            Ok(Self::default())
        }
    }

    /// Loads configuration from a specific file path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the YAML configuration file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The file contains invalid YAML
    /// - The YAML doesn't match the expected Config structure
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = Config::load_from_file(Path::new("./my-config.yaml"))?;
    /// ```
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Saves the configuration to a file.
    ///
    /// Creates parent directories if they don't exist.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the configuration should be saved
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Parent directories cannot be created
    /// - The file cannot be written
    /// - Serialization fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = Config::default();
    /// config.save(Path::new("./config.yaml"))?;
    /// ```
    pub fn save(&self, path: &Path) -> Result<()> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Returns the default configuration file path.
    ///
    /// # Platform-specific Paths
    ///
    /// - **Unix/macOS**: `~/.config/panpsychism/config.yaml`
    /// - **Windows**: `%APPDATA%/panpsychism/config.yaml`
    ///
    /// If the home directory cannot be determined, falls back to
    /// `./panpsychism-config.yaml` in the current directory.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let path = Config::config_path();
    /// println!("Config location: {:?}", path);
    /// ```
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .map(|p| p.join("panpsychism").join("config.yaml"))
            .unwrap_or_else(|| PathBuf::from("./panpsychism-config.yaml"))
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();

        assert_eq!(config.prompts_dir, PathBuf::from("./prompts"));
        assert_eq!(config.data_dir, PathBuf::from("./data"));
        assert_eq!(config.index_file, PathBuf::from("./data/masters.mv2"));
        assert_eq!(
            config.llm_endpoint,
            "http://127.0.0.1:8045/v1/chat/completions"
        );
        assert_eq!(config.llm_api_key, "sk-antigravity");
        assert_eq!(config.llm_model, "gemini-3-flash");
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let yaml = serde_yaml::to_string(&config).expect("Should serialize");

        assert!(yaml.contains("prompts_dir"));
        assert!(yaml.contains("llm_endpoint"));
        assert!(yaml.contains("gemini-3-flash"));
    }

    #[test]
    fn test_config_deserialization() {
        let yaml = r#"
prompts_dir: /custom/prompts
data_dir: /custom/data
index_file: /custom/index.mv2
llm_endpoint: http://localhost:8080/v1/chat/completions
llm_api_key: custom-key
llm_model: gpt-4
"#;

        let config: Config = serde_yaml::from_str(yaml).expect("Should deserialize");

        assert_eq!(config.prompts_dir, PathBuf::from("/custom/prompts"));
        assert_eq!(config.data_dir, PathBuf::from("/custom/data"));
        assert_eq!(config.llm_api_key, "custom-key");
        assert_eq!(config.llm_model, "gpt-4");
    }

    #[test]
    fn test_partial_config_uses_defaults() {
        let yaml = r#"
prompts_dir: /custom/prompts
"#;

        let config: Config = serde_yaml::from_str(yaml).expect("Should deserialize");

        assert_eq!(config.prompts_dir, PathBuf::from("/custom/prompts"));
        // Other fields should use defaults
        assert_eq!(config.data_dir, PathBuf::from("./data"));
        assert_eq!(config.llm_model, "gemini-3-flash");
    }

    #[test]
    fn test_load_from_file() {
        let mut temp_file = NamedTempFile::new().expect("Should create temp file");
        let yaml = r#"
prompts_dir: /test/prompts
llm_model: test-model
"#;
        temp_file.write_all(yaml.as_bytes()).expect("Should write");

        let config = Config::load_from_file(temp_file.path()).expect("Should load");

        assert_eq!(config.prompts_dir, PathBuf::from("/test/prompts"));
        assert_eq!(config.llm_model, "test-model");
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let temp_dir = tempfile::tempdir().expect("Should create temp dir");
        let config_path = temp_dir.path().join("config.yaml");

        let original = Config {
            prompts_dir: PathBuf::from("/roundtrip/prompts"),
            data_dir: PathBuf::from("/roundtrip/data"),
            index_file: PathBuf::from("/roundtrip/index.mv2"),
            llm_endpoint: "http://test:8080/v1/chat/completions".to_string(),
            llm_api_key: "test-key".to_string(),
            llm_model: "test-model".to_string(),
            privacy: PrivacyConfig::default(),
            analyze: None,
            output: None,
            content_mapping: None,
        };

        original.save(&config_path).expect("Should save");
        let loaded = Config::load_from_file(&config_path).expect("Should load");

        assert_eq!(original.prompts_dir, loaded.prompts_dir);
        assert_eq!(original.llm_endpoint, loaded.llm_endpoint);
        assert_eq!(original.llm_model, loaded.llm_model);
    }

    #[test]
    fn test_config_path_returns_valid_path() {
        let path = Config::config_path();

        // Should end with config.yaml
        assert!(path.to_string_lossy().ends_with("config.yaml"));
        // Should contain panpsychism directory
        assert!(path.to_string_lossy().contains("panpsychism"));
    }

    // =========================================================================
    // NEW CONFIG EXTENSION TESTS
    // =========================================================================

    #[test]
    fn test_analyze_config_default() {
        let analyze = AnalyzeConfig::default();

        assert_eq!(analyze.default_input, InputSource::Stdin);
        assert_eq!(analyze.default_outputs, vec![OutputDestination::Clipboard]);
        assert!(!analyze.verbose);
    }

    #[test]
    fn test_analyze_config_serialization_deserialization() {
        let yaml = r#"
default_input: clipboard
default_outputs:
  - stdout
  - secondcore
verbose: true
"#;

        let analyze: AnalyzeConfig = serde_yaml::from_str(yaml).expect("Should deserialize");

        assert_eq!(analyze.default_input, InputSource::Clipboard);
        assert_eq!(analyze.default_outputs.len(), 2);
        assert!(analyze.default_outputs.contains(&OutputDestination::Stdout));
        assert!(analyze.default_outputs.contains(&OutputDestination::SecondCore));
        assert!(analyze.verbose);

        // Roundtrip test
        let serialized = serde_yaml::to_string(&analyze).expect("Should serialize");
        let deserialized: AnalyzeConfig =
            serde_yaml::from_str(&serialized).expect("Should deserialize again");
        assert_eq!(analyze.default_input, deserialized.default_input);
        assert_eq!(analyze.verbose, deserialized.verbose);
    }

    #[test]
    fn test_output_config_default() {
        let output = OutputConfig::default();

        assert!(output.clipboard);
        assert!(output.file_path.is_none());
        assert!(!output.second_core);
        assert_eq!(
            output.second_core_endpoint,
            Some("https://webhook.mustafasarac.com".to_string())
        );
    }

    #[test]
    fn test_output_config_serialization_deserialization() {
        let yaml = r#"
clipboard: false
file_path: "/tmp/output.md"
second_core: true
second_core_endpoint: "https://custom.endpoint.com"
"#;

        let output: OutputConfig = serde_yaml::from_str(yaml).expect("Should deserialize");

        assert!(!output.clipboard);
        assert_eq!(output.file_path, Some("/tmp/output.md".to_string()));
        assert!(output.second_core);
        assert_eq!(
            output.second_core_endpoint,
            Some("https://custom.endpoint.com".to_string())
        );
    }

    #[test]
    fn test_content_type_mapping_default() {
        let mapping = ContentTypeMapping::default();

        assert_eq!(mapping.meeting, "prompts/master/meeting-notes.yaml");
        assert_eq!(mapping.code, "prompts/master/code-request.yaml");
        assert_eq!(mapping.blog, "prompts/master/blog-writing.yaml");
        assert_eq!(mapping.note, "prompts/master/daily-note.yaml");
        assert_eq!(mapping.household, "prompts/master/household.yaml");
    }

    #[test]
    fn test_content_type_mapping_serialization_deserialization() {
        let yaml = r#"
meeting: "custom/meeting.yaml"
code: "custom/code.yaml"
blog: "custom/blog.yaml"
note: "custom/note.yaml"
household: "custom/household.yaml"
"#;

        let mapping: ContentTypeMapping = serde_yaml::from_str(yaml).expect("Should deserialize");

        assert_eq!(mapping.meeting, "custom/meeting.yaml");
        assert_eq!(mapping.code, "custom/code.yaml");
        assert_eq!(mapping.blog, "custom/blog.yaml");
        assert_eq!(mapping.note, "custom/note.yaml");
        assert_eq!(mapping.household, "custom/household.yaml");
    }

    #[test]
    fn test_input_source_file_variant() {
        // File variant uses tagged format with !file
        let yaml = r#"
default_input: !file "/path/to/input.txt"
default_outputs: []
verbose: false
"#;

        let analyze: AnalyzeConfig = serde_yaml::from_str(yaml).expect("Should deserialize");

        match analyze.default_input {
            InputSource::File(path) => assert_eq!(path, "/path/to/input.txt"),
            _ => panic!("Expected File variant"),
        }
    }

    #[test]
    fn test_full_config_with_new_fields() {
        let yaml = r#"
prompts_dir: /custom/prompts
llm_model: gemini-3-flash

analyze:
  default_input: stdin
  default_outputs:
    - clipboard
    - file
  verbose: true

output:
  clipboard: true
  file_path: "/tmp/output.md"
  second_core: true
  second_core_endpoint: "https://webhook.mustafasarac.com"

content_mapping:
  meeting: "prompts/meeting.yaml"
  code: "prompts/code.yaml"
  blog: "prompts/blog.yaml"
  note: "prompts/note.yaml"
  household: "prompts/household.yaml"
"#;

        let config: Config = serde_yaml::from_str(yaml).expect("Should deserialize");

        // Check base config
        assert_eq!(config.prompts_dir, PathBuf::from("/custom/prompts"));
        assert_eq!(config.llm_model, "gemini-3-flash");

        // Check analyze config
        let analyze = config.analyze.expect("Should have analyze config");
        assert_eq!(analyze.default_input, InputSource::Stdin);
        assert!(analyze.verbose);
        assert_eq!(analyze.default_outputs.len(), 2);

        // Check output config
        let output = config.output.expect("Should have output config");
        assert!(output.clipboard);
        assert!(output.second_core);
        assert_eq!(output.file_path, Some("/tmp/output.md".to_string()));

        // Check content mapping
        let mapping = config.content_mapping.expect("Should have content mapping");
        assert_eq!(mapping.meeting, "prompts/meeting.yaml");
    }

    #[test]
    fn test_partial_new_config_uses_defaults() {
        let yaml = r#"
prompts_dir: /custom/prompts
analyze:
  verbose: true
"#;

        let config: Config = serde_yaml::from_str(yaml).expect("Should deserialize");

        // Base config should be set
        assert_eq!(config.prompts_dir, PathBuf::from("/custom/prompts"));

        // Analyze config should have partial override
        let analyze = config.analyze.expect("Should have analyze config");
        assert!(analyze.verbose);
        // But default_input should still be default
        assert_eq!(analyze.default_input, InputSource::Stdin);

        // Output and content_mapping should be None
        assert!(config.output.is_none());
        assert!(config.content_mapping.is_none());
    }
}
