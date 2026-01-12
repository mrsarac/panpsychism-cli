//! Error types for Project Panpsychism.
//!
//! This module provides comprehensive error handling with:
//! - Error codes for programmatic handling (E001-E099)
//! - User-friendly messages with actionable suggestions
//! - Structured error types using thiserror
//!
//! # Error Code Ranges
//!
//! | Range | Category |
//! |-------|----------|
//! | E001-E009 | Configuration errors |
//! | E010-E019 | Index errors |
//! | E020-E029 | Search errors |
//! | E030-E039 | API/Network errors |
//! | E040-E049 | Validation errors |
//! | E050-E059 | Synthesis errors |
//! | E060-E069 | Orchestration errors |
//! | E070-E079 | I/O errors |
//! | E080-E089 | Serialization errors |
//! | E090-E099 | Internal errors |
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::error::{Error, display_error};
//!
//! fn main() {
//!     let err = Error::config_file_not_found("./config.toml");
//!     display_error(&err);
//!     std::process::exit(err.exit_code());
//! }
//! ```
//!
//! ## Macros
//!
//! The [`bail_if!`] macro provides early return on condition:
//!
//! ```ignore
//! use panpsychism::error::{bail_if, Error};
//!
//! fn validate_input(value: i32) -> Result<()> {
//!     bail_if!(value < 0, Error::Validation("value must be non-negative".into()));
//!     Ok(())
//! }
//! ```

use thiserror::Error;

// =============================================================================
// CLI EXIT CODES
// =============================================================================

/// Exit code for successful execution.
pub const EXIT_SUCCESS: i32 = 0;

/// Exit code for configuration errors (missing files, invalid format, etc.).
pub const EXIT_CONFIG_ERROR: i32 = 1;

/// Exit code for index-related errors (missing index, corrupt index, etc.).
pub const EXIT_INDEX_ERROR: i32 = 2;

/// Exit code for search errors (empty query, no results, etc.).
pub const EXIT_SEARCH_ERROR: i32 = 3;

/// Exit code for API/network errors (connection failed, timeout, etc.).
pub const EXIT_API_ERROR: i32 = 4;

/// Exit code for validation errors (Spinoza validation failures).
pub const EXIT_VALIDATION_ERROR: i32 = 5;

/// Exit code for synthesis errors (prompt generation failures).
pub const EXIT_SYNTHESIS_ERROR: i32 = 6;

/// Exit code for I/O errors (file read/write failures).
pub const EXIT_IO_ERROR: i32 = 10;

/// Exit code for internal/unexpected errors.
pub const EXIT_INTERNAL_ERROR: i32 = 99;

// =============================================================================
// BAIL_IF MACRO
// =============================================================================

/// Early return if condition is true.
///
/// This macro simplifies conditional error returns, reducing boilerplate
/// for validation and guard clauses.
///
/// # Examples
///
/// ```ignore
/// use panpsychism::error::{bail_if, Error, Result};
///
/// fn process(items: &[String]) -> Result<()> {
///     bail_if!(items.is_empty(), Error::Validation("items cannot be empty".into()));
///     bail_if!(items.len() > 100, Error::Validation("too many items".into()));
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! bail_if {
    ($cond:expr, $err:expr) => {
        if $cond {
            return Err($err);
        }
    };
}

// Re-export for convenience
pub use bail_if;

// =============================================================================
// ERROR TYPE
// =============================================================================

/// The main error type for Project Panpsychism.
///
/// Each variant includes an error code prefix for easy identification
/// and programmatic handling. Use the `suggestion()` method to get
/// actionable guidance for resolving the error.
#[derive(Error, Debug)]
pub enum Error {
    // =========================================================================
    // CONFIGURATION ERRORS (E001-E009)
    // =========================================================================
    /// Configuration file not found.
    #[error("[E001] Configuration file not found: {path}")]
    ConfigFileNotFound {
        path: String,
        #[source]
        source: Option<std::io::Error>,
    },

    /// Configuration file has invalid format.
    #[error("[E002] Invalid configuration format in {path}: {details}")]
    ConfigInvalidFormat { path: String, details: String },

    /// Required configuration value is missing.
    #[error("[E003] Missing required configuration: {key}")]
    ConfigMissingKey { key: String },

    /// Configuration value is invalid.
    #[error("[E004] Invalid configuration value for '{key}': {details}")]
    ConfigInvalidValue { key: String, details: String },

    /// General configuration error (legacy compatibility).
    #[error("[E005] Configuration error: {0}")]
    Config(String),

    // =========================================================================
    // INDEX ERRORS (E010-E019)
    // =========================================================================
    /// Index file not found at expected location.
    #[error("[E010] Index not found at {path}. Run `panpsychism index` first.")]
    IndexNotFound { path: String },

    /// Index file is corrupted or unreadable.
    #[error("[E011] Index corrupted or unreadable: {path}")]
    IndexCorrupt { path: String, details: String },

    /// Prompts directory not found.
    #[error("[E012] Prompts directory not found: {path}")]
    PromptsDirectoryNotFound { path: String },

    /// Failed to parse prompt frontmatter.
    #[error("[E013] Invalid YAML frontmatter in {path}: {details}")]
    InvalidFrontmatter { path: String, details: String },

    /// General index error (legacy compatibility).
    #[error("[E014] Index error: {0}")]
    Index(String),

    // =========================================================================
    // SEARCH ERRORS (E020-E029)
    // =========================================================================
    /// Search query is empty or contains no valid terms.
    #[error("[E020] Empty or invalid search query")]
    SearchEmptyQuery,

    /// No results found for the search query.
    #[error("[E021] No results found for query: {query}")]
    SearchNoResults { query: String },

    /// Search filter is invalid (e.g., unknown category).
    #[error("[E022] Invalid search filter: {filter}")]
    SearchInvalidFilter { filter: String },

    /// General search error (legacy compatibility).
    #[error("[E023] Search error: {0}")]
    Search(String),

    // =========================================================================
    // API/NETWORK ERRORS (E030-E039)
    // =========================================================================
    /// API endpoint is unreachable.
    #[error("[E030] Cannot connect to API at {endpoint}")]
    ApiConnectionFailed {
        endpoint: String,
        #[source]
        source: Option<reqwest::Error>,
    },

    /// API request timed out.
    #[error("[E031] API request timed out after {timeout_secs}s")]
    ApiTimeout { timeout_secs: u64 },

    /// API returned an error response.
    #[error("[E032] API error ({status}): {message}")]
    ApiResponse { status: u16, message: String },

    /// API authentication failed.
    #[error("[E033] API authentication failed")]
    ApiAuthFailed,

    /// API rate limit exceeded.
    #[error("[E034] API rate limit exceeded")]
    ApiRateLimited { retry_after_secs: Option<u64> },

    /// HTTP client error.
    #[error("[E035] HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    // =========================================================================
    // VALIDATION ERRORS (E040-E049)
    // =========================================================================
    /// Content failed Spinoza validation.
    #[error(
        "[E040] Validation failed: {principle} score {score:.2} below threshold {threshold:.2}"
    )]
    ValidationFailed {
        principle: String,
        score: f64,
        threshold: f64,
    },

    /// Content is empty and cannot be validated.
    #[error("[E041] Cannot validate empty content")]
    ValidationEmptyContent,

    /// General validation error (legacy compatibility).
    #[error("[E042] Validation error: {0}")]
    Validation(String),

    // =========================================================================
    // SYNTHESIS ERRORS (E050-E059)
    // =========================================================================
    /// Prompt synthesis failed due to missing template.
    #[error("[E050] Synthesis template not found: {template}")]
    SynthesisTemplateNotFound { template: String },

    /// Prompt synthesis failed due to invalid parameters.
    #[error("[E051] Invalid synthesis parameters: {details}")]
    SynthesisInvalidParams { details: String },

    /// No response received from synthesis API.
    #[error("[E052] No response received from synthesis API")]
    SynthesisNoResponse,

    /// General synthesis error (legacy compatibility).
    #[error("[E053] Synthesis error: {0}")]
    Synthesis(String),

    // =========================================================================
    // ORCHESTRATION ERRORS (E060-E069)
    // =========================================================================
    /// Orchestration error.
    #[error("[E060] Orchestration error: {0}")]
    Orchestration(String),

    /// Correction/second-throw error.
    #[error("[E061] Correction error: {0}")]
    Correction(String),

    /// Privacy tier violation.
    #[error("[E062] Privacy error: {0}")]
    Privacy(String),

    // =========================================================================
    // I/O ERRORS (E070-E079)
    // =========================================================================
    /// File read error.
    #[error("[E070] Failed to read file: {path}")]
    FileReadError {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// File write error.
    #[error("[E071] Failed to write file: {path}")]
    FileWriteError {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Directory creation error.
    #[error("[E072] Failed to create directory: {path}")]
    DirectoryCreateError {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// General I/O error.
    #[error("[E073] I/O error: {0}")]
    Io(#[from] std::io::Error),

    // =========================================================================
    // SERIALIZATION ERRORS (E080-E089)
    // =========================================================================
    /// JSON serialization/deserialization error.
    #[error("[E080] JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// YAML serialization/deserialization error.
    #[error("[E081] YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// TOML serialization/deserialization error.
    #[error("[E082] TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    // =========================================================================
    // INTERNAL ERRORS (E090-E099)
    // =========================================================================
    /// An unexpected internal error occurred.
    #[error("[E090] Internal error: {message}")]
    Internal { message: String },

    /// A feature is not yet implemented.
    #[error("[E091] Not implemented: {feature}")]
    NotImplemented { feature: String },
}

// =============================================================================
// CONSTRUCTOR METHODS
// =============================================================================

impl Error {
    // -------------------------------------------------------------------------
    // Configuration errors
    // -------------------------------------------------------------------------

    /// Create a config file not found error.
    pub fn config_file_not_found(path: impl Into<String>) -> Self {
        Self::ConfigFileNotFound {
            path: path.into(),
            source: None,
        }
    }

    /// Create a config file not found error with source.
    pub fn config_file_not_found_with_source(
        path: impl Into<String>,
        source: std::io::Error,
    ) -> Self {
        Self::ConfigFileNotFound {
            path: path.into(),
            source: Some(source),
        }
    }

    /// Create an invalid config format error.
    pub fn config_invalid_format(path: impl Into<String>, details: impl Into<String>) -> Self {
        Self::ConfigInvalidFormat {
            path: path.into(),
            details: details.into(),
        }
    }

    /// Create a missing config key error.
    pub fn config_missing_key(key: impl Into<String>) -> Self {
        Self::ConfigMissingKey { key: key.into() }
    }

    // -------------------------------------------------------------------------
    // Index errors
    // -------------------------------------------------------------------------

    /// Create an index not found error.
    pub fn index_not_found(path: impl Into<String>) -> Self {
        Self::IndexNotFound { path: path.into() }
    }

    /// Create an index corrupt error.
    pub fn index_corrupt(path: impl Into<String>, details: impl Into<String>) -> Self {
        Self::IndexCorrupt {
            path: path.into(),
            details: details.into(),
        }
    }

    /// Create a prompts directory not found error.
    pub fn prompts_dir_not_found(path: impl Into<String>) -> Self {
        Self::PromptsDirectoryNotFound { path: path.into() }
    }

    /// Create an invalid frontmatter error.
    pub fn invalid_frontmatter(path: impl Into<String>, details: impl Into<String>) -> Self {
        Self::InvalidFrontmatter {
            path: path.into(),
            details: details.into(),
        }
    }

    // -------------------------------------------------------------------------
    // Search errors
    // -------------------------------------------------------------------------

    /// Create a search no results error.
    pub fn search_no_results(query: impl Into<String>) -> Self {
        Self::SearchNoResults {
            query: query.into(),
        }
    }

    /// Create a search invalid filter error.
    pub fn search_invalid_filter(filter: impl Into<String>) -> Self {
        Self::SearchInvalidFilter {
            filter: filter.into(),
        }
    }

    // -------------------------------------------------------------------------
    // API errors
    // -------------------------------------------------------------------------

    /// Create an API connection failed error.
    pub fn api_connection_failed(endpoint: impl Into<String>) -> Self {
        Self::ApiConnectionFailed {
            endpoint: endpoint.into(),
            source: None,
        }
    }

    /// Create an API connection failed error with source.
    pub fn api_connection_failed_with_source(
        endpoint: impl Into<String>,
        source: reqwest::Error,
    ) -> Self {
        Self::ApiConnectionFailed {
            endpoint: endpoint.into(),
            source: Some(source),
        }
    }

    /// Create an API timeout error.
    pub fn api_timeout(timeout_secs: u64) -> Self {
        Self::ApiTimeout { timeout_secs }
    }

    /// Create an API response error.
    pub fn api_response(status: u16, message: impl Into<String>) -> Self {
        Self::ApiResponse {
            status,
            message: message.into(),
        }
    }

    // -------------------------------------------------------------------------
    // Validation errors
    // -------------------------------------------------------------------------

    /// Create a validation failed error.
    pub fn validation_failed(principle: impl Into<String>, score: f64, threshold: f64) -> Self {
        Self::ValidationFailed {
            principle: principle.into(),
            score,
            threshold,
        }
    }

    // -------------------------------------------------------------------------
    // Synthesis errors
    // -------------------------------------------------------------------------

    /// Create a synthesis template not found error.
    pub fn synthesis_template_not_found(template: impl Into<String>) -> Self {
        Self::SynthesisTemplateNotFound {
            template: template.into(),
        }
    }

    /// Create a synthesis invalid params error.
    pub fn synthesis_invalid_params(details: impl Into<String>) -> Self {
        Self::SynthesisInvalidParams {
            details: details.into(),
        }
    }

    // -------------------------------------------------------------------------
    // I/O errors
    // -------------------------------------------------------------------------

    /// Create a file read error.
    pub fn file_read_error(path: impl Into<String>, source: std::io::Error) -> Self {
        Self::FileReadError {
            path: path.into(),
            source,
        }
    }

    /// Create a file write error.
    pub fn file_write_error(path: impl Into<String>, source: std::io::Error) -> Self {
        Self::FileWriteError {
            path: path.into(),
            source,
        }
    }

    // -------------------------------------------------------------------------
    // Internal errors
    // -------------------------------------------------------------------------

    /// Create an internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Create a not implemented error.
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
        }
    }
}

// =============================================================================
// ERROR METADATA
// =============================================================================

impl Error {
    /// Get the error code (e.g., "E001").
    pub fn code(&self) -> &'static str {
        match self {
            // Configuration
            Self::ConfigFileNotFound { .. } => "E001",
            Self::ConfigInvalidFormat { .. } => "E002",
            Self::ConfigMissingKey { .. } => "E003",
            Self::ConfigInvalidValue { .. } => "E004",
            Self::Config(_) => "E005",

            // Index
            Self::IndexNotFound { .. } => "E010",
            Self::IndexCorrupt { .. } => "E011",
            Self::PromptsDirectoryNotFound { .. } => "E012",
            Self::InvalidFrontmatter { .. } => "E013",
            Self::Index(_) => "E014",

            // Search
            Self::SearchEmptyQuery => "E020",
            Self::SearchNoResults { .. } => "E021",
            Self::SearchInvalidFilter { .. } => "E022",
            Self::Search(_) => "E023",

            // API
            Self::ApiConnectionFailed { .. } => "E030",
            Self::ApiTimeout { .. } => "E031",
            Self::ApiResponse { .. } => "E032",
            Self::ApiAuthFailed => "E033",
            Self::ApiRateLimited { .. } => "E034",
            Self::Http(_) => "E035",

            // Validation
            Self::ValidationFailed { .. } => "E040",
            Self::ValidationEmptyContent => "E041",
            Self::Validation(_) => "E042",

            // Synthesis
            Self::SynthesisTemplateNotFound { .. } => "E050",
            Self::SynthesisInvalidParams { .. } => "E051",
            Self::SynthesisNoResponse => "E052",
            Self::Synthesis(_) => "E053",

            // Orchestration
            Self::Orchestration(_) => "E060",
            Self::Correction(_) => "E061",
            Self::Privacy(_) => "E062",

            // I/O
            Self::FileReadError { .. } => "E070",
            Self::FileWriteError { .. } => "E071",
            Self::DirectoryCreateError { .. } => "E072",
            Self::Io(_) => "E073",

            // Serialization
            Self::Json(_) => "E080",
            Self::Yaml(_) => "E081",
            Self::Toml(_) => "E082",

            // Internal
            Self::Internal { .. } => "E090",
            Self::NotImplemented { .. } => "E091",
        }
    }

    /// Get a suggestion for how to resolve the error.
    pub fn suggestion(&self) -> Option<&'static str> {
        match self {
            // Configuration
            Self::ConfigFileNotFound { .. } => Some(
                "Run `panpsychism init` to create a default configuration file, \
                 or specify the config path with --config",
            ),
            Self::ConfigInvalidFormat { .. } => Some(
                "Check the configuration file syntax. TOML format is required. \
                 Use `panpsychism config --validate` to check your config",
            ),
            Self::ConfigMissingKey { .. } => Some(
                "Add the missing key to your configuration file. \
                 See `panpsychism config --example` for a complete example",
            ),
            Self::ConfigInvalidValue { .. } => {
                Some("Check the value format and valid options in the documentation")
            }
            Self::Config(_) => Some("Check your configuration file for errors"),

            // Index
            Self::IndexNotFound { .. } => {
                Some("Run `panpsychism index` to build the search index first")
            }
            Self::IndexCorrupt { .. } => {
                Some("Delete the index file and run `panpsychism index` to rebuild it")
            }
            Self::PromptsDirectoryNotFound { .. } => Some(
                "Create the prompts directory or update the `prompts_dir` setting in your config",
            ),
            Self::InvalidFrontmatter { .. } => Some(
                "Check the YAML frontmatter syntax in the specified file. \
                 Required fields: `title`. Optional: `description`, `tags`, `category`",
            ),
            Self::Index(_) => Some("Try rebuilding the index with `panpsychism index --force`"),

            // Search
            Self::SearchEmptyQuery => {
                Some("Provide a search query, e.g., `panpsychism search \"authentication\"`")
            }
            Self::SearchNoResults { .. } => Some(
                "Try a broader search query, different keywords, or check available tags with \
                 `panpsychism list --tags`",
            ),
            Self::SearchInvalidFilter { .. } => {
                Some("Use `panpsychism list --categories` to see available categories")
            }
            Self::Search(_) => Some("Check your search parameters and try again"),

            // API
            Self::ApiConnectionFailed { .. } => Some(
                "Check that Antigravity proxy is running at http://127.0.0.1:8045, \
                 or configure a different endpoint with --endpoint",
            ),
            Self::ApiTimeout { .. } => Some(
                "Try increasing the timeout with --timeout, \
                 or check your network connection",
            ),
            Self::ApiResponse { status, .. } if *status == 401 || *status == 403 => Some(
                "Check your API key is correct. Set GEMINI_API_KEY environment variable \
                 or use --api-key",
            ),
            Self::ApiResponse { status, .. } if *status == 429 => Some(
                "Rate limit exceeded. Wait a moment and try again, \
                 or reduce request frequency",
            ),
            Self::ApiResponse { status, .. } if *status >= 500 => {
                Some("The API server is experiencing issues. Try again later")
            }
            Self::ApiResponse { .. } => Some("Check the API documentation for this status code"),
            Self::ApiAuthFailed => Some("Check your API key is correct and has not expired"),
            Self::ApiRateLimited { .. } => {
                Some("Wait for the rate limit window to reset, then try again")
            }
            Self::Http(_) => Some("Check your network connection and try again"),

            // Validation
            Self::ValidationFailed { .. } => Some(
                "Review your content for the failing principle. \
                 Use `panpsychism validate --verbose` for detailed feedback",
            ),
            Self::ValidationEmptyContent => Some("Provide non-empty content for validation"),
            Self::Validation(_) => Some("Check the validation parameters and try again"),

            // Synthesis
            Self::SynthesisTemplateNotFound { .. } => {
                Some("Use `panpsychism templates --list` to see available templates")
            }
            Self::SynthesisInvalidParams { .. } => {
                Some("Check the template documentation for required parameters")
            }
            Self::SynthesisNoResponse => Some("Check your API connection and try again"),
            Self::Synthesis(_) => Some("Check the synthesis parameters and try again"),

            // Orchestration
            Self::Orchestration(_) => {
                Some("Check the orchestration configuration and prompt availability")
            }
            Self::Correction(_) => {
                Some("The self-correction phase failed. Try adjusting your query")
            }
            Self::Privacy(_) => Some("Check the privacy tier of the requested prompt"),

            // I/O
            Self::FileReadError { .. } => {
                Some("Check that the file exists and you have read permissions")
            }
            Self::FileWriteError { .. } => {
                Some("Check that you have write permissions for the target directory")
            }
            Self::DirectoryCreateError { .. } => {
                Some("Check that you have permissions to create directories in the target path")
            }
            Self::Io(_) => Some("Check file permissions and disk space"),

            // Serialization
            Self::Json(_) => Some("Check the JSON syntax for errors"),
            Self::Yaml(_) => {
                Some("Check the YAML syntax for errors. Use a YAML validator if needed")
            }
            Self::Toml(_) => {
                Some("Check the TOML syntax for errors. Use a TOML validator if needed")
            }

            // Internal
            Self::Internal { .. } => Some(
                "This is an internal error. Please report it at \
                 https://github.com/mrsarac/prompt-library/issues",
            ),
            Self::NotImplemented { .. } => Some("This feature is planned but not yet available"),
        }
    }

    /// Get the appropriate CLI exit code for this error.
    pub fn exit_code(&self) -> i32 {
        match self {
            // Configuration
            Self::ConfigFileNotFound { .. }
            | Self::ConfigInvalidFormat { .. }
            | Self::ConfigMissingKey { .. }
            | Self::ConfigInvalidValue { .. }
            | Self::Config(_) => EXIT_CONFIG_ERROR,

            // Index
            Self::IndexNotFound { .. }
            | Self::IndexCorrupt { .. }
            | Self::PromptsDirectoryNotFound { .. }
            | Self::InvalidFrontmatter { .. }
            | Self::Index(_) => EXIT_INDEX_ERROR,

            // Search
            Self::SearchEmptyQuery
            | Self::SearchNoResults { .. }
            | Self::SearchInvalidFilter { .. }
            | Self::Search(_) => EXIT_SEARCH_ERROR,

            // API
            Self::ApiConnectionFailed { .. }
            | Self::ApiTimeout { .. }
            | Self::ApiResponse { .. }
            | Self::ApiAuthFailed
            | Self::ApiRateLimited { .. }
            | Self::Http(_) => EXIT_API_ERROR,

            // Validation
            Self::ValidationFailed { .. } | Self::ValidationEmptyContent | Self::Validation(_) => {
                EXIT_VALIDATION_ERROR
            }

            // Synthesis
            Self::SynthesisTemplateNotFound { .. }
            | Self::SynthesisInvalidParams { .. }
            | Self::SynthesisNoResponse
            | Self::Synthesis(_) => EXIT_SYNTHESIS_ERROR,

            // Orchestration (use synthesis exit code)
            Self::Orchestration(_) | Self::Correction(_) | Self::Privacy(_) => EXIT_SYNTHESIS_ERROR,

            // I/O
            Self::FileReadError { .. }
            | Self::FileWriteError { .. }
            | Self::DirectoryCreateError { .. }
            | Self::Io(_) => EXIT_IO_ERROR,

            // Serialization (treated as I/O)
            Self::Json(_) | Self::Yaml(_) | Self::Toml(_) => EXIT_IO_ERROR,

            // Internal
            Self::Internal { .. } | Self::NotImplemented { .. } => EXIT_INTERNAL_ERROR,
        }
    }

    /// Check if this error is recoverable (can be retried).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::ApiTimeout { .. } | Self::ApiRateLimited { .. } | Self::Http(_)
        ) || matches!(self, Self::ApiResponse { status, .. } if *status >= 500)
    }
}

// =============================================================================
// ERROR DISPLAY UTILITIES
// =============================================================================

/// Display an error in a user-friendly format with suggestions.
///
/// Outputs to stderr with formatting:
/// ```text
/// Error [E001]: Configuration file not found: ./config.toml
///
/// Suggestion: Run `panpsychism init` to create a default configuration file,
///             or specify the config path with --config
/// ```
pub fn display_error(err: &Error) {
    eprintln!("\n\x1b[1;31mError:\x1b[0m {}\n", err);

    if let Some(suggestion) = err.suggestion() {
        eprintln!("\x1b[1;33mSuggestion:\x1b[0m {}\n", suggestion);
    }
}

/// Display an error without colors (for non-TTY output).
pub fn display_error_plain(err: &Error) {
    eprintln!("\nError: {}\n", err);

    if let Some(suggestion) = err.suggestion() {
        eprintln!("Suggestion: {}\n", suggestion);
    }
}

/// Display an error with full details including source chain.
pub fn display_error_verbose(err: &Error) {
    eprintln!("\n\x1b[1;31mError:\x1b[0m [{}] {}\n", err.code(), err);

    // Display source chain if available
    if let Some(source) = std::error::Error::source(err) {
        eprintln!("\x1b[1;90mCaused by:\x1b[0m {}\n", source);
    }

    if let Some(suggestion) = err.suggestion() {
        eprintln!("\x1b[1;33mSuggestion:\x1b[0m {}\n", suggestion);
    }

    eprintln!("\x1b[1;90mHint:\x1b[0m For more information, run with RUST_BACKTRACE=1\n");
}

// =============================================================================
// RESULT TYPE ALIAS
// =============================================================================

/// A Result type alias using our Error type.
pub type Result<T> = std::result::Result<T, Error>;

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(Error::config_file_not_found("test.toml").code(), "E001");
        assert_eq!(Error::index_not_found("index.mv2").code(), "E010");
        assert_eq!(Error::SearchEmptyQuery.code(), "E020");
        assert_eq!(
            Error::api_connection_failed("http://localhost").code(),
            "E030"
        );
        assert_eq!(Error::ValidationEmptyContent.code(), "E041");
    }

    #[test]
    fn test_error_suggestions() {
        let err = Error::index_not_found("./index.mv2");
        assert!(err.suggestion().is_some());
        assert!(err.suggestion().unwrap().contains("panpsychism index"));
    }

    #[test]
    fn test_exit_codes() {
        assert_eq!(
            Error::config_file_not_found("test").exit_code(),
            EXIT_CONFIG_ERROR
        );
        assert_eq!(Error::index_not_found("test").exit_code(), EXIT_INDEX_ERROR);
        assert_eq!(Error::SearchEmptyQuery.exit_code(), EXIT_SEARCH_ERROR);
        assert_eq!(
            Error::api_connection_failed("test").exit_code(),
            EXIT_API_ERROR
        );
        assert_eq!(
            Error::ValidationEmptyContent.exit_code(),
            EXIT_VALIDATION_ERROR
        );
    }

    #[test]
    fn test_is_recoverable() {
        assert!(Error::api_timeout(30).is_recoverable());
        assert!(Error::ApiRateLimited {
            retry_after_secs: Some(60)
        }
        .is_recoverable());
        assert!(!Error::config_file_not_found("test").is_recoverable());
        assert!(!Error::ValidationEmptyContent.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let err = Error::config_file_not_found("./missing.toml");
        let display = format!("{}", err);
        assert!(display.contains("E001"));
        assert!(display.contains("Configuration file not found"));
        assert!(display.contains("missing.toml"));
    }

    #[test]
    fn test_legacy_error_variants() {
        // Ensure legacy variants still work
        let err = Error::Config("legacy error".to_string());
        assert_eq!(err.code(), "E005");

        let err = Error::Index("legacy index error".to_string());
        assert_eq!(err.code(), "E014");

        let err = Error::Search("legacy search error".to_string());
        assert_eq!(err.code(), "E023");
    }

    #[test]
    fn test_api_response_suggestions() {
        let err_401 = Error::api_response(401, "Unauthorized");
        assert!(err_401.suggestion().unwrap().contains("API key"));

        let err_429 = Error::api_response(429, "Too many requests");
        assert!(err_429.suggestion().unwrap().contains("Rate limit"));

        let err_500 = Error::api_response(500, "Server error");
        assert!(err_500.suggestion().unwrap().contains("server"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }

    #[test]
    fn test_result_type_alias() {
        fn example_fn() -> Result<i32> {
            Ok(42)
        }
        assert_eq!(example_fn().unwrap(), 42);
    }

    #[test]
    fn test_bail_if_returns_error_on_true() {
        fn check_positive(n: i32) -> Result<()> {
            bail_if!(n < 0, Error::Validation("must be non-negative".into()));
            Ok(())
        }

        assert!(check_positive(-1).is_err());
        assert!(check_positive(0).is_ok());
        assert!(check_positive(1).is_ok());
    }

    #[test]
    fn test_bail_if_continues_on_false() {
        fn validate_range(n: i32) -> Result<i32> {
            bail_if!(n < 0, Error::Validation("too small".into()));
            bail_if!(n > 100, Error::Validation("too large".into()));
            Ok(n * 2)
        }

        assert_eq!(validate_range(50).unwrap(), 100);
        assert!(validate_range(-1).is_err());
        assert!(validate_range(101).is_err());
    }
}
