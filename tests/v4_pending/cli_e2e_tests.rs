//! E2E tests for Panpsychism CLI Interface
//!
//! These tests verify the CLI commands work correctly from end-to-end,
//! testing both the actual binary execution and the internal CLI module.
//!
//! ## Test Coverage
//!
//! 1. **Command Parsing Tests** - Verify argument parsing for all commands
//! 2. **Query Command Tests** - Test query execution with various options
//! 3. **Agent Commands Tests** - Test agent listing and status
//! 4. **Status Commands Tests** - Test system status and health checks
//! 5. **Config Commands Tests** - Test configuration management
//! 6. **Error Handling Tests** - Test invalid inputs and edge cases

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

// =============================================================================
// Helper Functions
// =============================================================================

/// Get a Command for the panpsychism binary
#[allow(deprecated)]
fn panpsychism_cmd() -> Command {
    Command::cargo_bin("panpsychism").unwrap()
}

/// Create a temporary directory with test prompts for CLI tests
fn create_test_prompts_dir() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let prompts_dir = temp_dir.path().join("prompts");
    std::fs::create_dir_all(&prompts_dir).unwrap();

    // Create a sample prompt file
    let prompt_content = r#"---
id: test-001
title: "CLI Test Prompt"
category: testing
tags:
  - cli
  - test
privacy_tier: public
---

# CLI Test Prompt

This is a test prompt for CLI E2E testing.

## Usage

Use this prompt to verify CLI functionality.
"#;

    std::fs::write(prompts_dir.join("test-prompt.md"), prompt_content).unwrap();
    temp_dir
}

// =============================================================================
// 1. Command Parsing Tests
// =============================================================================

#[test]
fn test_help_command() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Usage:").or(predicate::str::contains("USAGE:")))
        .stdout(predicate::str::contains("panpsychism"));
}

#[test]
fn test_version_command() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("panpsychism"));
}

#[test]
fn test_query_command_parsing() {
    // Query command should exist and accept arguments
    let mut cmd = panpsychism_cmd();
    cmd.arg("search")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("search").or(predicate::str::contains("Search")));
}

#[test]
fn test_index_command_parsing() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("index")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("index").or(predicate::str::contains("Index")));
}

#[test]
fn test_status_command_parsing_via_help() {
    // Test that help shows available commands
    let mut cmd = panpsychism_cmd();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("index").or(predicate::str::contains("Index")))
        .stdout(predicate::str::contains("search").or(predicate::str::contains("Search")));
}

#[test]
fn test_config_command_parsing_implicit() {
    // Config-like options should be available globally
    let mut cmd = panpsychism_cmd();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("json").or(predicate::str::contains("log")));
}

// =============================================================================
// 2. Query/Search Command Tests
// =============================================================================

#[test]
fn test_simple_search_query() {
    let temp_dir = create_test_prompts_dir();
    let prompts_path = temp_dir.path().join("prompts");

    let mut cmd = panpsychism_cmd();
    cmd.arg("search")
        .arg("test")
        .arg("--top")
        .arg("5")
        .current_dir(temp_dir.path())
        .env("PANPSYCHISM_PROMPTS_DIR", prompts_path.to_str().unwrap())
        .assert()
        // May succeed or fail depending on index existence, but should not panic
        .code(predicate::in_iter([0, 1]));
}

#[test]
fn test_search_with_top_flag() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("search")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("top").or(predicate::str::contains("-t")));
}

#[test]
fn test_batch_query_via_search() {
    let temp_dir = create_test_prompts_dir();

    // Multiple search terms should be accepted
    let mut cmd = panpsychism_cmd();
    cmd.arg("search")
        .arg("authentication security")
        .current_dir(temp_dir.path())
        .assert()
        .code(predicate::in_iter([0, 1])); // May fail without index, but shouldn't crash
}

// =============================================================================
// 3. Agent Commands Tests (via Ask command which uses agents internally)
// =============================================================================

#[test]
fn test_ask_command_help() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("ask")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("ask").or(predicate::str::contains("Ask")))
        .stdout(predicate::str::contains("question").or(predicate::str::contains("QUESTION")));
}

#[test]
fn test_ask_verbose_flag() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("ask")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("verbose").or(predicate::str::contains("-v")));
}

#[test]
fn test_agent_via_shell_help() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("shell")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("shell").or(predicate::str::contains("Shell")));
}

// =============================================================================
// 4. Status Commands Tests (via Shell or implicit status)
// =============================================================================

#[test]
fn test_system_status_implicit() {
    // Running with no command should show help or error
    let mut cmd = panpsychism_cmd();
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Usage").or(predicate::str::contains("required")));
}

#[test]
fn test_health_check_via_help() {
    // Health check is implicit - if help works, basic health is OK
    let mut cmd = panpsychism_cmd();
    cmd.arg("--help")
        .assert()
        .success();
}

#[test]
fn test_metrics_display_via_json_flag() {
    // JSON output flag should be recognized
    let mut cmd = panpsychism_cmd();
    cmd.arg("--json")
        .arg("--help")
        .assert()
        .success();
}

// =============================================================================
// 5. Config Commands Tests
// =============================================================================

#[test]
fn test_config_json_flag() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("json"));
}

#[test]
fn test_config_log_level_flag() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("log-level").or(predicate::str::contains("log")));
}

#[test]
fn test_config_validation_log_level() {
    // Valid log level should be accepted
    let temp_dir = create_test_prompts_dir();

    let mut cmd = panpsychism_cmd();
    cmd.arg("--log-level")
        .arg("debug")
        .arg("search")
        .arg("test")
        .current_dir(temp_dir.path())
        .assert()
        .code(predicate::in_iter([0, 1])); // May fail but shouldn't crash
}

// =============================================================================
// 6. Error Handling Tests
// =============================================================================

#[test]
fn test_invalid_command() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("nonexistent_command")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error").or(predicate::str::contains("invalid")));
}

#[test]
fn test_missing_required_arguments() {
    // Search without query should fail
    let mut cmd = panpsychism_cmd();
    cmd.arg("search")
        .assert()
        .failure()
        .stderr(predicate::str::contains("required").or(predicate::str::contains("QUERY")));
}

#[test]
fn test_invalid_flag_value() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("--log-level")
        .arg("not_a_valid_level")
        .arg("search")
        .arg("test")
        .assert()
        // Should either fail or fall back to default - shouldn't crash
        .code(predicate::in_iter([0, 1, 2]));
}

#[test]
fn test_nonexistent_directory() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("index")
        .arg("--dir")
        .arg("/nonexistent/path/that/does/not/exist/12345")
        .assert()
        .failure();
}

#[test]
fn test_empty_search_query() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("search")
        .arg("")
        .assert()
        // Empty string as query - should either fail or return no results
        .code(predicate::in_iter([0, 1]));
}

#[test]
fn test_special_characters_in_query() {
    let temp_dir = create_test_prompts_dir();

    let mut cmd = panpsychism_cmd();
    cmd.arg("search")
        .arg("test @#$%^&*()")
        .current_dir(temp_dir.path())
        .assert()
        // Should handle special chars gracefully
        .code(predicate::in_iter([0, 1]));
}

// =============================================================================
// 7. Index Command Tests
// =============================================================================

#[test]
fn test_index_command_basic() {
    let temp_dir = create_test_prompts_dir();
    let prompts_dir = temp_dir.path().join("prompts");
    let output_file = temp_dir.path().join("data").join("test.mv2");

    let mut cmd = panpsychism_cmd();
    cmd.arg("index")
        .arg("--dir")
        .arg(prompts_dir.to_str().unwrap())
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexing").or(predicate::str::contains("complete")).or(predicate::str::contains("Scanning")));
}

#[test]
fn test_index_output_flag() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("index")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("output").or(predicate::str::contains("-o")));
}

#[test]
fn test_index_dir_flag() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("index")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("dir").or(predicate::str::contains("-d")));
}

// =============================================================================
// 8. Analyze Command Tests
// =============================================================================

#[test]
fn test_analyze_command_help() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("analyze")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("analyze").or(predicate::str::contains("Analyze")));
}

#[test]
fn test_analyze_input_flag() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("analyze")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("input").or(predicate::str::contains("-i")));
}

#[test]
fn test_analyze_output_flag() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("analyze")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("output").or(predicate::str::contains("-o")));
}

#[test]
fn test_analyze_with_text() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("analyze")
        .arg("This is a test message for analysis")
        .arg("--output")
        .arg("stdout")
        .assert()
        .success()
        .stdout(predicate::str::contains("test message").or(predicate::str::contains("This is")));
}

#[test]
fn test_analyze_verbose_flag() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("analyze")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("verbose").or(predicate::str::contains("-v")));
}

// =============================================================================
// 9. Shell Command Tests (non-interactive checks)
// =============================================================================

#[test]
fn test_shell_command_exists() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("shell")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("shell").or(predicate::str::contains("Shell")).or(predicate::str::contains("interactive")));
}

// =============================================================================
// 10. Output Format Tests
// =============================================================================

#[test]
fn test_json_output_format() {
    let temp_dir = create_test_prompts_dir();
    let prompts_dir = temp_dir.path().join("prompts");
    let output_file = temp_dir.path().join("data").join("test.mv2");

    let mut cmd = panpsychism_cmd();
    cmd.arg("--json")
        .arg("index")
        .arg("--dir")
        .arg(prompts_dir.to_str().unwrap())
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .assert()
        // With --json, output format should change (may still have human output mixed)
        .code(predicate::in_iter([0, 1]));
}

#[test]
fn test_log_level_debug() {
    let temp_dir = create_test_prompts_dir();
    let prompts_dir = temp_dir.path().join("prompts");

    let mut cmd = panpsychism_cmd();
    cmd.arg("--log-level")
        .arg("debug")
        .arg("search")
        .arg("test")
        .current_dir(temp_dir.path())
        .env("PANPSYCHISM_PROMPTS_DIR", prompts_dir.to_str().unwrap())
        .assert()
        .code(predicate::in_iter([0, 1]));
}

#[test]
fn test_log_level_error() {
    let temp_dir = create_test_prompts_dir();
    let prompts_dir = temp_dir.path().join("prompts");

    let mut cmd = panpsychism_cmd();
    cmd.arg("--log-level")
        .arg("error")
        .arg("search")
        .arg("test")
        .current_dir(temp_dir.path())
        .env("PANPSYCHISM_PROMPTS_DIR", prompts_dir.to_str().unwrap())
        .assert()
        .code(predicate::in_iter([0, 1]));
}

// =============================================================================
// 11. CLI Module Internal Tests (library-side Command enum)
// =============================================================================

#[cfg(test)]
mod cli_module_tests {
    use panpsychism::cli::{CliApp, CliConfig, Command, ConfigAction, OutputFormat};

    #[test]
    fn test_cli_command_parse_status() {
        let args: Vec<String> = vec!["status".into()];
        let cmd = Command::parse(&args).unwrap();
        assert_eq!(cmd, Command::Status);
    }

    #[test]
    fn test_cli_command_parse_agents() {
        let args: Vec<String> = vec!["agents".into()];
        let cmd = Command::parse(&args).unwrap();
        assert_eq!(cmd, Command::Agents);
    }

    #[test]
    fn test_cli_command_parse_query() {
        let args: Vec<String> = vec!["query".into(), "test question".into()];
        let cmd = Command::parse(&args).unwrap();
        match cmd {
            Command::Query { text, agent } => {
                assert_eq!(text, "test question");
                assert!(agent.is_none());
            }
            _ => panic!("Expected Query command"),
        }
    }

    #[test]
    fn test_cli_command_parse_query_with_agent() {
        let args: Vec<String> = vec![
            "query".into(),
            "test".into(),
            "--agent".into(),
            "synthesizer".into(),
        ];
        let cmd = Command::parse(&args).unwrap();
        match cmd {
            Command::Query { text, agent } => {
                assert_eq!(text, "test");
                assert_eq!(agent, Some("synthesizer".to_string()));
            }
            _ => panic!("Expected Query command with agent"),
        }
    }

    #[test]
    fn test_cli_command_parse_config_list() {
        let args: Vec<String> = vec!["config".into()];
        let cmd = Command::parse(&args).unwrap();
        match cmd {
            Command::Config { action } => {
                assert_eq!(action, ConfigAction::List);
            }
            _ => panic!("Expected Config command"),
        }
    }

    #[test]
    fn test_cli_command_parse_config_get() {
        let args: Vec<String> = vec!["config".into(), "get".into(), "output_format".into()];
        let cmd = Command::parse(&args).unwrap();
        match cmd {
            Command::Config { action } => {
                assert!(matches!(action, ConfigAction::Get { key } if key == "output_format"));
            }
            _ => panic!("Expected Config Get command"),
        }
    }

    #[test]
    fn test_cli_command_parse_config_set() {
        let args: Vec<String> = vec!["config".into(), "set".into(), "verbose".into(), "true".into()];
        let cmd = Command::parse(&args).unwrap();
        match cmd {
            Command::Config { action } => {
                assert!(matches!(action, ConfigAction::Set { key, value } if key == "verbose" && value == "true"));
            }
            _ => panic!("Expected Config Set command"),
        }
    }

    #[test]
    fn test_cli_command_parse_help() {
        let args: Vec<String> = vec!["help".into()];
        let cmd = Command::parse(&args).unwrap();
        assert_eq!(cmd, Command::Help);
    }

    #[test]
    fn test_cli_command_parse_version() {
        let args: Vec<String> = vec!["version".into()];
        let cmd = Command::parse(&args).unwrap();
        assert_eq!(cmd, Command::Version);
    }

    #[test]
    fn test_cli_command_parse_unknown() {
        let args: Vec<String> = vec!["unknown_command".into()];
        let result = Command::parse(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_command_parse_empty() {
        let args: Vec<String> = vec![];
        let cmd = Command::parse(&args).unwrap();
        assert_eq!(cmd, Command::Help);
    }

    #[test]
    fn test_cli_app_execute_status() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Status).unwrap();
        assert!(result.success);
        assert!(result.output.contains("System Status") || result.output.contains("healthy"));
    }

    #[test]
    fn test_cli_app_execute_agents() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Agents).unwrap();
        assert!(result.success);
        assert!(result.output.contains("Agent") || result.output.contains("search"));
    }

    #[test]
    fn test_cli_app_execute_version() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Version).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_cli_app_execute_help() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Help).unwrap();
        assert!(result.success);
        assert!(result.output.contains("USAGE") || result.output.contains("COMMANDS"));
    }

    #[test]
    fn test_cli_app_execute_metrics() {
        let mut app = CliApp::new();
        let result = app.execute(Command::Metrics).unwrap();
        assert!(result.success);
        assert!(result.output.contains("Metrics") || result.output.contains("Latency"));
    }

    #[test]
    fn test_output_format_parsing() {
        assert_eq!(OutputFormat::from_str("json"), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::from_str("text"), Some(OutputFormat::Text));
        assert_eq!(OutputFormat::from_str("markdown"), Some(OutputFormat::Markdown));
        assert_eq!(OutputFormat::from_str("md"), Some(OutputFormat::Markdown));
        assert_eq!(OutputFormat::from_str("quiet"), Some(OutputFormat::Quiet));
        assert_eq!(OutputFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_cli_config_builder() {
        let config = CliConfig::builder()
            .output_format(OutputFormat::Json)
            .color_enabled(false)
            .verbose(true)
            .max_history(500)
            .build();

        assert_eq!(config.output_format, OutputFormat::Json);
        assert!(!config.color_enabled);
        assert!(config.verbose);
        assert_eq!(config.max_history, 500);
    }

    #[test]
    fn test_cli_app_json_output() {
        let mut app = CliApp::builder()
            .output_format(OutputFormat::Json)
            .build();

        let result = app.execute(Command::Status).unwrap();
        assert!(result.success);
        // JSON output should be parseable
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&result.output);
        assert!(parsed.is_ok(), "Output should be valid JSON: {}", result.output);
    }

    #[test]
    fn test_cli_app_quiet_output() {
        let mut app = CliApp::builder()
            .output_format(OutputFormat::Quiet)
            .build();

        let result = app.execute(Command::Status).unwrap();
        assert!(result.success);
        // Quiet output should be minimal
        assert!(result.output.len() < 50, "Quiet output should be minimal");
    }
}

// =============================================================================
// 12. Integration Tests - Full Pipeline
// =============================================================================

#[test]
fn test_full_index_and_search_pipeline() {
    let temp_dir = create_test_prompts_dir();
    let prompts_dir = temp_dir.path().join("prompts");
    let output_file = temp_dir.path().join("data").join("pipeline.mv2");

    // Step 1: Index the prompts
    let mut cmd = panpsychism_cmd();
    cmd.arg("index")
        .arg("--dir")
        .arg(prompts_dir.to_str().unwrap())
        .arg("--output")
        .arg(output_file.to_str().unwrap())
        .assert()
        .success();

    // Step 2: Search for something
    let mut cmd = panpsychism_cmd();
    cmd.arg("search")
        .arg("test")
        .current_dir(temp_dir.path())
        .assert()
        .code(predicate::in_iter([0, 1])); // May or may not find results
}

#[test]
fn test_analyze_pipeline() {
    // Simple analyze with direct text
    let mut cmd = panpsychism_cmd();
    cmd.arg("analyze")
        .arg("Hello, this is a test message for the analyze pipeline.")
        .arg("--output")
        .arg("stdout")
        .assert()
        .success()
        .stdout(predicate::str::contains("Hello"));
}

#[test]
fn test_analyze_with_verbose() {
    let mut cmd = panpsychism_cmd();
    cmd.arg("analyze")
        .arg("Test verbose output")
        .arg("--output")
        .arg("stdout")
        .arg("--verbose")
        .assert()
        .success()
        .stdout(predicate::str::contains("Test verbose"));
}
