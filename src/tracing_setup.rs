//! Tracing and logging setup for Project Panpsychism.
//!
//! Provides structured logging with configurable output formats (pretty or JSON)
//! and environment-based log level filtering.
//!
//! # Example
//!
//! ```rust,ignore
//! use panpsychism::tracing_setup::setup_logging;
//!
//! // Human-readable output (default)
//! setup_logging(false, "info");
//!
//! // JSON output for machine parsing
//! setup_logging(true, "debug");
//! ```

use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    prelude::*,
    EnvFilter,
};

/// Initialize the tracing subscriber with configurable format.
///
/// # Arguments
///
/// * `json` - If true, output logs in JSON format (for machine parsing).
///   If false, use human-readable pretty format.
/// * `default_level` - Default log level if RUST_LOG is not set.
///   Options: "error", "warn", "info", "debug", "trace"
///
/// # Environment Variables
///
/// - `RUST_LOG`: Override log level filter (e.g., "debug", "panpsychism=trace")
/// - `PANPSYCHISM_LOG_JSON`: Set to "1" or "true" to enable JSON output
///
/// # Example
///
/// ```rust,ignore
/// // Pretty output with info level
/// setup_logging(false, "info");
///
/// // JSON output with debug level
/// setup_logging(true, "debug");
/// ```
pub fn setup_logging(json: bool, default_level: &str) {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));

    if json {
        let subscriber = tracing_subscriber::registry().with(env_filter).with(
            fmt::layer()
                .json()
                .with_span_events(FmtSpan::CLOSE)
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true),
        );
        subscriber.init();
    } else {
        let subscriber = tracing_subscriber::registry().with(env_filter).with(
            fmt::layer()
                .pretty()
                .with_target(true)
                .with_thread_ids(false)
                .with_file(false)
                .with_line_number(false),
        );
        subscriber.init();
    }
}

/// Check if JSON logging is requested via environment variable.
pub fn should_use_json() -> bool {
    std::env::var("PANPSYCHISM_LOG_JSON")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_use_json_default() {
        // Clean environment for test
        std::env::remove_var("PANPSYCHISM_LOG_JSON");
        assert!(!should_use_json());
    }

    #[test]
    fn test_should_use_json_enabled() {
        std::env::set_var("PANPSYCHISM_LOG_JSON", "1");
        assert!(should_use_json());
        std::env::remove_var("PANPSYCHISM_LOG_JSON");
    }
}
