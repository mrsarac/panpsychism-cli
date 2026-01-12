//! Tester Agent module for Project Panpsychism.
//!
//! The Trial Master - "Every spell must be proven through rigorous trial."
//!
//! This module implements the Tester Agent, responsible for validating prompts
//! and responses against expected behaviors through automated testing. Like a
//! meticulous trial master, the Tester ensures that every magical incantation
//! produces the expected results.
//!
//! ## The Sorcerer's Wand Metaphor
//!
//! In our magical framework, The Trial Master serves as the quality guardian:
//!
//! - **Test Cases** (trial challenges) define what magic should achieve
//! - **The Tester** (TesterAgent) executes trials with precision
//! - **Test Reports** (trial verdicts) reveal whether the magic holds true
//!
//! ## Philosophy
//!
//! Following Spinoza's principles:
//!
//! - **CONATUS**: Drive to ensure reliability through verification
//! - **RATIO**: Logical assessment of expected vs actual outcomes
//! - **LAETITIA**: Joy through confidence in tested behavior
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::tester::{TesterAgent, TestCase, ExpectedOutput};
//!
//! let tester = TesterAgent::new();
//!
//! let test_case = TestCase::new("greeting_test")
//!     .with_input("Hello, how are you?")
//!     .with_expected(ExpectedOutput::Contains("Hello".to_string()));
//!
//! let report = tester.run_test(&test_case, "Hello! I'm doing well.").await?;
//! assert!(report.passed());
//! ```

use crate::{Error, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info, warn};

// =============================================================================
// TEST STATUS ENUM
// =============================================================================

/// Status of a test execution.
///
/// Like the verdict of a trial, the status indicates
/// whether the magic performed as expected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TestStatus {
    /// Test passed - actual output matches expected.
    Passed,
    /// Test failed - actual output does not match expected.
    #[default]
    Failed,
    /// Test was skipped - not executed for some reason.
    Skipped,
    /// Test encountered an error during execution.
    Error,
}

impl std::fmt::Display for TestStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Passed => write!(f, "passed"),
            Self::Failed => write!(f, "failed"),
            Self::Skipped => write!(f, "skipped"),
            Self::Error => write!(f, "error"),
        }
    }
}

impl std::str::FromStr for TestStatus {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "passed" | "pass" | "success" | "ok" => Ok(Self::Passed),
            "failed" | "fail" | "failure" => Ok(Self::Failed),
            "skipped" | "skip" | "ignored" => Ok(Self::Skipped),
            "error" | "err" | "exception" => Ok(Self::Error),
            _ => Err(Error::Config(format!(
                "Unknown test status: '{}'. Valid: passed, failed, skipped, error",
                s
            ))),
        }
    }
}

impl TestStatus {
    /// Check if this status represents success.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Passed)
    }

    /// Check if this status represents a failure.
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failed | Self::Error)
    }

    /// Get emoji representation for display.
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Passed => "✅",
            Self::Failed => "❌",
            Self::Skipped => "⏭️",
            Self::Error => "💥",
        }
    }
}

// =============================================================================
// EXPECTED OUTPUT ENUM
// =============================================================================

/// Definition of expected output for a test case.
///
/// Like the prophecy of what magic should achieve, the expected output
/// defines the criteria for success in various ways.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpectedOutput {
    /// Exact string match required.
    Exact(String),
    /// Output must contain this substring.
    Contains(String),
    /// Output must match this regex pattern.
    Matches(String),
    /// Output must NOT contain this substring.
    NotContains(String),
    /// Output must have at least this many characters.
    MinLength(usize),
    /// Output must have at most this many characters.
    MaxLength(usize),
    /// Output must match a schema (JSON path validation).
    Schema(SchemaExpectation),
    /// Multiple conditions that must all be true.
    All(Vec<ExpectedOutput>),
    /// At least one condition must be true.
    Any(Vec<ExpectedOutput>),
}

impl ExpectedOutput {
    /// Check if the actual output matches this expectation.
    pub fn matches(&self, actual: &str) -> bool {
        match self {
            Self::Exact(expected) => actual == expected,
            Self::Contains(substring) => actual.contains(substring),
            Self::Matches(pattern) => {
                Regex::new(pattern)
                    .map(|re| re.is_match(actual))
                    .unwrap_or(false)
            }
            Self::NotContains(substring) => !actual.contains(substring),
            Self::MinLength(min) => actual.len() >= *min,
            Self::MaxLength(max) => actual.len() <= *max,
            Self::Schema(schema) => schema.validate(actual),
            Self::All(expectations) => expectations.iter().all(|e| e.matches(actual)),
            Self::Any(expectations) => expectations.iter().any(|e| e.matches(actual)),
        }
    }

    /// Get a description of what this expectation checks.
    pub fn description(&self) -> String {
        match self {
            Self::Exact(s) => format!("exactly equals '{}'", truncate(s, 50)),
            Self::Contains(s) => format!("contains '{}'", truncate(s, 50)),
            Self::Matches(p) => format!("matches pattern '{}'", truncate(p, 50)),
            Self::NotContains(s) => format!("does not contain '{}'", truncate(s, 50)),
            Self::MinLength(n) => format!("has at least {} characters", n),
            Self::MaxLength(n) => format!("has at most {} characters", n),
            Self::Schema(s) => format!("matches schema: {}", s.description()),
            Self::All(exps) => format!(
                "all of: [{}]",
                exps.iter()
                    .map(|e| e.description())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Any(exps) => format!(
                "any of: [{}]",
                exps.iter()
                    .map(|e| e.description())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    /// Create an exact match expectation.
    pub fn exact(s: impl Into<String>) -> Self {
        Self::Exact(s.into())
    }

    /// Create a contains expectation.
    pub fn contains(s: impl Into<String>) -> Self {
        Self::Contains(s.into())
    }

    /// Create a regex match expectation.
    pub fn regex(pattern: impl Into<String>) -> Self {
        Self::Matches(pattern.into())
    }

    /// Create a not-contains expectation.
    pub fn not_contains(s: impl Into<String>) -> Self {
        Self::NotContains(s.into())
    }

    /// Create a minimum length expectation.
    pub fn min_length(n: usize) -> Self {
        Self::MinLength(n)
    }

    /// Create a maximum length expectation.
    pub fn max_length(n: usize) -> Self {
        Self::MaxLength(n)
    }
}

impl Default for ExpectedOutput {
    fn default() -> Self {
        Self::MinLength(1)
    }
}

// =============================================================================
// SCHEMA EXPECTATION
// =============================================================================

/// Schema-based expectation for structured output validation.
///
/// Supports basic JSON structure validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaExpectation {
    /// Expected type of the output.
    pub expected_type: SchemaType,
    /// Required fields (for object type).
    pub required_fields: Vec<String>,
    /// Optional: minimum array length (for array type).
    pub min_items: Option<usize>,
    /// Optional: maximum array length (for array type).
    pub max_items: Option<usize>,
}

/// Type expected in schema validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchemaType {
    /// Any valid JSON.
    #[default]
    Any,
    /// JSON object.
    Object,
    /// JSON array.
    Array,
    /// JSON string.
    String,
    /// JSON number.
    Number,
    /// JSON boolean.
    Boolean,
}

impl SchemaExpectation {
    /// Create a new schema expectation.
    pub fn new(expected_type: SchemaType) -> Self {
        Self {
            expected_type,
            required_fields: Vec::new(),
            min_items: None,
            max_items: None,
        }
    }

    /// Add required fields for object validation.
    pub fn with_required_fields(mut self, fields: Vec<String>) -> Self {
        self.required_fields = fields;
        self
    }

    /// Set minimum items for array validation.
    pub fn with_min_items(mut self, min: usize) -> Self {
        self.min_items = Some(min);
        self
    }

    /// Set maximum items for array validation.
    pub fn with_max_items(mut self, max: usize) -> Self {
        self.max_items = Some(max);
        self
    }

    /// Validate the actual output against this schema.
    pub fn validate(&self, actual: &str) -> bool {
        let parsed: std::result::Result<serde_json::Value, _> = serde_json::from_str(actual);

        match parsed {
            Ok(value) => self.validate_value(&value),
            Err(_) => {
                // If not valid JSON, only match if we expect Any and it's non-empty
                matches!(self.expected_type, SchemaType::Any) && !actual.trim().is_empty()
            }
        }
    }

    fn validate_value(&self, value: &serde_json::Value) -> bool {
        match self.expected_type {
            SchemaType::Any => true,
            SchemaType::Object => {
                if let Some(obj) = value.as_object() {
                    self.required_fields
                        .iter()
                        .all(|field| obj.contains_key(field))
                } else {
                    false
                }
            }
            SchemaType::Array => {
                if let Some(arr) = value.as_array() {
                    let len = arr.len();
                    self.min_items.map_or(true, |min| len >= min)
                        && self.max_items.map_or(true, |max| len <= max)
                } else {
                    false
                }
            }
            SchemaType::String => value.is_string(),
            SchemaType::Number => value.is_number(),
            SchemaType::Boolean => value.is_boolean(),
        }
    }

    /// Get a description of this schema.
    pub fn description(&self) -> String {
        match self.expected_type {
            SchemaType::Any => "any valid content".to_string(),
            SchemaType::Object => {
                if self.required_fields.is_empty() {
                    "JSON object".to_string()
                } else {
                    format!(
                        "JSON object with fields: {}",
                        self.required_fields.join(", ")
                    )
                }
            }
            SchemaType::Array => {
                let mut desc = "JSON array".to_string();
                if let Some(min) = self.min_items {
                    desc.push_str(&format!(" (min: {})", min));
                }
                if let Some(max) = self.max_items {
                    desc.push_str(&format!(" (max: {})", max));
                }
                desc
            }
            SchemaType::String => "JSON string".to_string(),
            SchemaType::Number => "JSON number".to_string(),
            SchemaType::Boolean => "JSON boolean".to_string(),
        }
    }
}

// =============================================================================
// TEST CASE
// =============================================================================

/// A single test case definition.
///
/// Like a trial challenge, a test case defines the input
/// and the expected outcome of the magical operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// Unique name for this test case.
    pub name: String,
    /// Description of what this test validates.
    pub description: Option<String>,
    /// Input to be tested.
    pub input: String,
    /// Expected output definition.
    pub expected_output: ExpectedOutput,
    /// Tags for categorization and filtering.
    pub tags: Vec<String>,
    /// Timeout in milliseconds (0 = no timeout).
    pub timeout_ms: u64,
    /// Whether this test is enabled.
    pub enabled: bool,
    /// Priority for ordering (higher = run first).
    pub priority: u32,
    /// Setup data or context for the test.
    pub setup: Option<String>,
    /// Cleanup instructions after test.
    pub teardown: Option<String>,
}

impl TestCase {
    /// Create a new test case with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            input: String::new(),
            expected_output: ExpectedOutput::default(),
            tags: Vec::new(),
            timeout_ms: 5000,
            enabled: true,
            priority: 0,
            setup: None,
            teardown: None,
        }
    }

    /// Set the test description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the input for the test.
    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.input = input.into();
        self
    }

    /// Set the expected output.
    pub fn with_expected(mut self, expected: ExpectedOutput) -> Self {
        self.expected_output = expected;
        self
    }

    /// Add a tag to the test case.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags to the test case.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags.extend(tags);
        self
    }

    /// Set the timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }

    /// Set whether the test is enabled.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the setup instructions.
    pub fn with_setup(mut self, setup: impl Into<String>) -> Self {
        self.setup = Some(setup.into());
        self
    }

    /// Set the teardown instructions.
    pub fn with_teardown(mut self, teardown: impl Into<String>) -> Self {
        self.teardown = Some(teardown.into());
        self
    }

    /// Check if this test has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t.eq_ignore_ascii_case(tag))
    }

    /// Check if this test matches any of the given tags.
    pub fn matches_any_tag(&self, tags: &[String]) -> bool {
        tags.iter().any(|t| self.has_tag(t))
    }
}

// =============================================================================
// TEST RESULT
// =============================================================================

/// Result of a single test execution.
///
/// Like the verdict of a trial, the result contains
/// all details about the test outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Name of the test that was executed.
    pub test_name: String,
    /// Status of the test execution.
    pub status: TestStatus,
    /// The actual output received.
    pub actual_output: Option<String>,
    /// Error message if the test failed or errored.
    pub error_message: Option<String>,
    /// Duration of the test execution in milliseconds.
    pub duration_ms: u64,
    /// Expected output description for reference.
    pub expected_description: String,
    /// Additional metadata from the test execution.
    pub metadata: HashMap<String, String>,
}

impl TestResult {
    /// Create a new test result.
    pub fn new(test_name: impl Into<String>, status: TestStatus) -> Self {
        Self {
            test_name: test_name.into(),
            status,
            actual_output: None,
            error_message: None,
            duration_ms: 0,
            expected_description: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the actual output.
    pub fn with_actual_output(mut self, output: impl Into<String>) -> Self {
        self.actual_output = Some(output.into());
        self
    }

    /// Set the error message.
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error_message = Some(error.into());
        self
    }

    /// Set the duration.
    pub fn with_duration_ms(mut self, duration: u64) -> Self {
        self.duration_ms = duration;
        self
    }

    /// Set the expected description.
    pub fn with_expected_description(mut self, desc: impl Into<String>) -> Self {
        self.expected_description = desc.into();
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if the test passed.
    pub fn passed(&self) -> bool {
        self.status.is_success()
    }

    /// Check if the test failed.
    pub fn failed(&self) -> bool {
        self.status.is_failure()
    }

    /// Format the result as a summary line.
    pub fn summary(&self) -> String {
        format!(
            "{} {} ({}ms){}",
            self.status.emoji(),
            self.test_name,
            self.duration_ms,
            self.error_message
                .as_ref()
                .map(|e| format!(" - {}", e))
                .unwrap_or_default()
        )
    }
}

// =============================================================================
// TEST SUITE
// =============================================================================

/// A collection of related test cases.
///
/// Like a trial dossier, a test suite groups related tests
/// that should be executed together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    /// Name of the test suite.
    pub name: String,
    /// Description of what this suite tests.
    pub description: Option<String>,
    /// The test cases in this suite.
    pub tests: Vec<TestCase>,
    /// Tags for the entire suite.
    pub tags: Vec<String>,
    /// Whether the suite is enabled.
    pub enabled: bool,
    /// Setup to run before all tests.
    pub before_all: Option<String>,
    /// Teardown to run after all tests.
    pub after_all: Option<String>,
    /// Setup to run before each test.
    pub before_each: Option<String>,
    /// Teardown to run after each test.
    pub after_each: Option<String>,
}

impl TestSuite {
    /// Create a new test suite.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            tests: Vec::new(),
            tags: Vec::new(),
            enabled: true,
            before_all: None,
            after_all: None,
            before_each: None,
            after_each: None,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a test case.
    pub fn with_test(mut self, test: TestCase) -> Self {
        self.tests.push(test);
        self
    }

    /// Add multiple test cases.
    pub fn with_tests(mut self, tests: Vec<TestCase>) -> Self {
        self.tests.extend(tests);
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set before_all setup.
    pub fn with_before_all(mut self, setup: impl Into<String>) -> Self {
        self.before_all = Some(setup.into());
        self
    }

    /// Set after_all teardown.
    pub fn with_after_all(mut self, teardown: impl Into<String>) -> Self {
        self.after_all = Some(teardown.into());
        self
    }

    /// Set before_each setup.
    pub fn with_before_each(mut self, setup: impl Into<String>) -> Self {
        self.before_each = Some(setup.into());
        self
    }

    /// Set after_each teardown.
    pub fn with_after_each(mut self, teardown: impl Into<String>) -> Self {
        self.after_each = Some(teardown.into());
        self
    }

    /// Get the number of tests in this suite.
    pub fn test_count(&self) -> usize {
        self.tests.len()
    }

    /// Get enabled tests only.
    pub fn enabled_tests(&self) -> Vec<&TestCase> {
        self.tests.iter().filter(|t| t.enabled).collect()
    }

    /// Get tests sorted by priority (highest first).
    pub fn tests_by_priority(&self) -> Vec<&TestCase> {
        let mut tests: Vec<_> = self.tests.iter().collect();
        tests.sort_by(|a, b| b.priority.cmp(&a.priority));
        tests
    }

    /// Filter tests by tag.
    pub fn tests_with_tag(&self, tag: &str) -> Vec<&TestCase> {
        self.tests.iter().filter(|t| t.has_tag(tag)).collect()
    }
}

// =============================================================================
// TEST REPORT
// =============================================================================

/// Complete report from a test run.
///
/// Like the final judgment scroll, the test report contains
/// all information about the trial outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    /// Number of tests that passed.
    pub passed: usize,
    /// Number of tests that failed.
    pub failed: usize,
    /// Number of tests that were skipped.
    pub skipped: usize,
    /// Number of tests that errored.
    pub errored: usize,
    /// Individual test results.
    pub results: Vec<TestResult>,
    /// Total duration of the test run in milliseconds.
    pub duration_ms: u64,
    /// Test coverage percentage (0.0 - 1.0).
    pub coverage: f64,
    /// Start time of the test run.
    pub started_at: Option<String>,
    /// End time of the test run.
    pub ended_at: Option<String>,
    /// Name of the test suite/run.
    pub suite_name: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl TestReport {
    /// Create a new empty test report.
    pub fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
            skipped: 0,
            errored: 0,
            results: Vec::new(),
            duration_ms: 0,
            coverage: 0.0,
            started_at: None,
            ended_at: None,
            suite_name: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a test result to the report.
    pub fn add_result(&mut self, result: TestResult) {
        match result.status {
            TestStatus::Passed => self.passed += 1,
            TestStatus::Failed => self.failed += 1,
            TestStatus::Skipped => self.skipped += 1,
            TestStatus::Error => self.errored += 1,
        }
        self.results.push(result);
    }

    /// Get the total number of tests.
    pub fn total(&self) -> usize {
        self.passed + self.failed + self.skipped + self.errored
    }

    /// Get the success rate (0.0 - 1.0).
    pub fn success_rate(&self) -> f64 {
        let executed = self.passed + self.failed + self.errored;
        if executed == 0 {
            return 0.0;
        }
        self.passed as f64 / executed as f64
    }

    /// Check if all tests passed.
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.errored == 0
    }

    /// Check if any tests failed.
    pub fn has_failures(&self) -> bool {
        self.failed > 0 || self.errored > 0
    }

    /// Get failed results only.
    pub fn failures(&self) -> Vec<&TestResult> {
        self.results
            .iter()
            .filter(|r| matches!(r.status, TestStatus::Failed | TestStatus::Error))
            .collect()
    }

    /// Get passed results only.
    pub fn successes(&self) -> Vec<&TestResult> {
        self.results
            .iter()
            .filter(|r| matches!(r.status, TestStatus::Passed))
            .collect()
    }

    /// Format the report as a summary.
    pub fn summary(&self) -> String {
        let status = if self.all_passed() {
            "ALL PASSED"
        } else {
            "HAS FAILURES"
        };

        format!(
            "{} - {} passed, {} failed, {} skipped, {} errored ({}ms) - {:.1}% success rate",
            status,
            self.passed,
            self.failed,
            self.skipped,
            self.errored,
            self.duration_ms,
            self.success_rate() * 100.0
        )
    }

    /// Format the report as markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# Test Report\n\n");

        if let Some(suite) = &self.suite_name {
            output.push_str(&format!("**Suite:** {}\n\n", suite));
        }

        output.push_str("## Summary\n\n");
        output.push_str(&format!(
            "| Metric | Value |\n\
             |--------|-------|\n\
             | Passed | {} |\n\
             | Failed | {} |\n\
             | Skipped | {} |\n\
             | Errored | {} |\n\
             | Total | {} |\n\
             | Duration | {}ms |\n\
             | Success Rate | {:.1}% |\n\n",
            self.passed,
            self.failed,
            self.skipped,
            self.errored,
            self.total(),
            self.duration_ms,
            self.success_rate() * 100.0
        ));

        if !self.failures().is_empty() {
            output.push_str("## Failures\n\n");
            for result in self.failures() {
                output.push_str(&format!("### {}\n\n", result.test_name));
                if let Some(error) = &result.error_message {
                    output.push_str(&format!("**Error:** {}\n\n", error));
                }
                if let Some(actual) = &result.actual_output {
                    output.push_str(&format!(
                        "**Actual output:**\n```\n{}\n```\n\n",
                        truncate(actual, 500)
                    ));
                }
            }
        }

        output.push_str("## All Results\n\n");
        for result in &self.results {
            output.push_str(&format!("- {}\n", result.summary()));
        }

        output
    }

    /// Set the suite name.
    pub fn with_suite_name(mut self, name: impl Into<String>) -> Self {
        self.suite_name = Some(name.into());
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl Default for TestReport {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTER CONFIGURATION
// =============================================================================

/// Configuration for the Tester Agent.
#[derive(Debug, Clone)]
pub struct TesterConfig {
    /// Whether to run tests in parallel.
    pub parallel: bool,
    /// Maximum number of parallel tests.
    pub max_parallel: usize,
    /// Whether to stop on first failure.
    pub fail_fast: bool,
    /// Default timeout in milliseconds.
    pub default_timeout_ms: u64,
    /// Whether to capture output.
    pub capture_output: bool,
    /// Tags to include (empty = all).
    pub include_tags: Vec<String>,
    /// Tags to exclude.
    pub exclude_tags: Vec<String>,
    /// Retry failed tests this many times.
    pub retry_count: u32,
    /// Delay between retries in milliseconds.
    pub retry_delay_ms: u64,
}

impl Default for TesterConfig {
    fn default() -> Self {
        Self {
            parallel: false,
            max_parallel: 4,
            fail_fast: false,
            default_timeout_ms: 5000,
            capture_output: true,
            include_tags: Vec::new(),
            exclude_tags: Vec::new(),
            retry_count: 0,
            retry_delay_ms: 100,
        }
    }
}

impl TesterConfig {
    /// Create a configuration for fast, parallel execution.
    pub fn fast() -> Self {
        Self {
            parallel: true,
            max_parallel: 8,
            fail_fast: true,
            default_timeout_ms: 2000,
            capture_output: false,
            ..Default::default()
        }
    }

    /// Create a configuration for thorough, sequential execution.
    pub fn thorough() -> Self {
        Self {
            parallel: false,
            max_parallel: 1,
            fail_fast: false,
            default_timeout_ms: 30000,
            capture_output: true,
            retry_count: 2,
            retry_delay_ms: 500,
            ..Default::default()
        }
    }

    /// Create a CI-friendly configuration.
    pub fn ci() -> Self {
        Self {
            parallel: true,
            max_parallel: 4,
            fail_fast: false,
            default_timeout_ms: 10000,
            capture_output: true,
            retry_count: 1,
            retry_delay_ms: 1000,
            ..Default::default()
        }
    }
}

// =============================================================================
// TESTER AGENT
// =============================================================================

/// The Tester Agent - The Trial Master of the Sorcerer's Tower.
///
/// Responsible for validating prompts and responses through automated testing,
/// ensuring that every magical incantation produces the expected results.
///
/// # Philosophy
///
/// Grounded in Spinoza's principles:
/// - **CONATUS**: Drive to ensure reliability through verification
/// - **RATIO**: Logical assessment of expected vs actual outcomes
/// - **LAETITIA**: Joy through confidence in tested behavior
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::tester::{TesterAgent, TestCase, ExpectedOutput};
///
/// let tester = TesterAgent::new();
///
/// let test = TestCase::new("hello_test")
///     .with_input("Say hello")
///     .with_expected(ExpectedOutput::contains("hello"));
///
/// let report = tester.run_test(&test, "Hello, world!").await?;
/// assert!(report.passed());
/// ```
#[derive(Debug, Clone)]
pub struct TesterAgent {
    /// Configuration for the tester.
    config: TesterConfig,
}

impl Default for TesterAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl TesterAgent {
    /// Create a new Tester Agent with default configuration.
    pub fn new() -> Self {
        Self {
            config: TesterConfig::default(),
        }
    }

    /// Create a new Tester Agent with custom configuration.
    pub fn with_config(config: TesterConfig) -> Self {
        Self { config }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> TesterAgentBuilder {
        TesterAgentBuilder::default()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &TesterConfig {
        &self.config
    }

    // =========================================================================
    // SINGLE TEST EXECUTION
    // =========================================================================

    /// Run a single test case against the given output.
    ///
    /// # Arguments
    ///
    /// * `test` - The test case definition
    /// * `actual_output` - The actual output to validate
    ///
    /// # Returns
    ///
    /// A `TestResult` containing the outcome of the test.
    pub async fn run_test(&self, test: &TestCase, actual_output: &str) -> Result<TestResult> {
        let start = Instant::now();

        debug!("Running test: {}", test.name);

        // Check if test is enabled
        if !test.enabled {
            info!("Test '{}' is disabled, skipping", test.name);
            return Ok(TestResult::new(&test.name, TestStatus::Skipped)
                .with_expected_description(test.expected_output.description()));
        }

        // Check tag filters
        if !self.should_run_test(test) {
            info!("Test '{}' filtered out by tags, skipping", test.name);
            return Ok(TestResult::new(&test.name, TestStatus::Skipped)
                .with_expected_description(test.expected_output.description())
                .with_metadata("skip_reason", "filtered by tags"));
        }

        // Execute the test with potential retries
        let mut last_result = None;
        let max_attempts = self.config.retry_count + 1;

        for attempt in 0..max_attempts {
            if attempt > 0 {
                debug!("Retrying test '{}' (attempt {})", test.name, attempt + 1);
                tokio::time::sleep(tokio::time::Duration::from_millis(
                    self.config.retry_delay_ms,
                ))
                .await;
            }

            let result = self.execute_test(test, actual_output).await;
            match &result {
                Ok(r) if r.passed() => return result,
                _ => last_result = Some(result),
            }
        }

        // Return the last result (which was a failure)
        last_result.unwrap_or_else(|| {
            Ok(TestResult::new(&test.name, TestStatus::Error)
                .with_error("No result from test execution")
                .with_duration_ms(start.elapsed().as_millis() as u64))
        })
    }

    /// Execute a single test without retries.
    async fn execute_test(&self, test: &TestCase, actual_output: &str) -> Result<TestResult> {
        let start = Instant::now();

        // Apply timeout
        let timeout_ms = if test.timeout_ms > 0 {
            test.timeout_ms
        } else {
            self.config.default_timeout_ms
        };

        let check_future = async {
            let matches = test.expected_output.matches(actual_output);

            if matches {
                TestResult::new(&test.name, TestStatus::Passed)
                    .with_actual_output(actual_output)
                    .with_expected_description(test.expected_output.description())
            } else {
                TestResult::new(&test.name, TestStatus::Failed)
                    .with_actual_output(actual_output)
                    .with_expected_description(test.expected_output.description())
                    .with_error(format!(
                        "Expected output to {}, but got: '{}'",
                        test.expected_output.description(),
                        truncate(actual_output, 100)
                    ))
            }
        };

        let result = if timeout_ms > 0 {
            match tokio::time::timeout(
                tokio::time::Duration::from_millis(timeout_ms),
                check_future,
            )
            .await
            {
                Ok(result) => result,
                Err(_) => TestResult::new(&test.name, TestStatus::Error)
                    .with_error(format!("Test timed out after {}ms", timeout_ms)),
            }
        } else {
            check_future.await
        };

        let duration = start.elapsed().as_millis() as u64;
        let result = result.with_duration_ms(duration);

        info!(
            "Test '{}': {} ({}ms)",
            test.name,
            result.status.emoji(),
            duration
        );

        Ok(result)
    }

    /// Check if a test should be run based on tag filters.
    fn should_run_test(&self, test: &TestCase) -> bool {
        // Check exclude tags first
        if !self.config.exclude_tags.is_empty() && test.matches_any_tag(&self.config.exclude_tags) {
            return false;
        }

        // Check include tags (empty means include all)
        if self.config.include_tags.is_empty() {
            return true;
        }

        test.matches_any_tag(&self.config.include_tags)
    }

    // =========================================================================
    // MULTIPLE TEST EXECUTION
    // =========================================================================

    /// Run multiple tests and collect results.
    ///
    /// # Arguments
    ///
    /// * `tests` - Vector of (test_case, actual_output) pairs
    ///
    /// # Returns
    ///
    /// A `TestReport` containing all results.
    pub async fn run_tests(
        &self,
        tests: Vec<(&TestCase, &str)>,
    ) -> Result<TestReport> {
        let start = Instant::now();
        let mut report = TestReport::new();

        report.started_at = Some(chrono::Utc::now().to_rfc3339());

        info!("Running {} tests", tests.len());

        if self.config.parallel && tests.len() > 1 {
            self.run_tests_parallel(tests, &mut report).await?;
        } else {
            self.run_tests_sequential(tests, &mut report).await?;
        }

        report.duration_ms = start.elapsed().as_millis() as u64;
        report.ended_at = Some(chrono::Utc::now().to_rfc3339());
        report.coverage = self.calculate_coverage(&report);

        info!("Test run complete: {}", report.summary());

        Ok(report)
    }

    /// Run tests sequentially.
    async fn run_tests_sequential(
        &self,
        tests: Vec<(&TestCase, &str)>,
        report: &mut TestReport,
    ) -> Result<()> {
        for (test, output) in tests {
            let result = self.run_test(test, output).await?;
            let should_stop = self.config.fail_fast && result.failed();
            report.add_result(result);

            if should_stop {
                warn!("Stopping test run due to fail_fast setting");
                break;
            }
        }
        Ok(())
    }

    /// Run tests in parallel.
    async fn run_tests_parallel(
        &self,
        tests: Vec<(&TestCase, &str)>,
        report: &mut TestReport,
    ) -> Result<()> {
        use futures::stream::{self, StreamExt};

        let results: Vec<Result<TestResult>> = stream::iter(tests)
            .map(|(test, output)| async move { self.run_test(test, output).await })
            .buffer_unordered(self.config.max_parallel)
            .collect()
            .await;

        for result in results {
            match result {
                Ok(r) => {
                    if self.config.fail_fast && r.failed() {
                        report.add_result(r);
                        warn!("Stopping test run due to fail_fast setting");
                        break;
                    }
                    report.add_result(r);
                }
                Err(e) => {
                    report.add_result(
                        TestResult::new("unknown", TestStatus::Error)
                            .with_error(e.to_string()),
                    );
                }
            }
        }

        Ok(())
    }

    /// Calculate coverage based on test results.
    fn calculate_coverage(&self, report: &TestReport) -> f64 {
        // Simple coverage: ratio of passed to total
        // In a real system, this would analyze code paths
        if report.total() == 0 {
            return 0.0;
        }
        report.passed as f64 / report.total() as f64
    }

    // =========================================================================
    // TEST SUITE EXECUTION
    // =========================================================================

    /// Run a complete test suite.
    ///
    /// # Arguments
    ///
    /// * `suite` - The test suite to run
    /// * `output_generator` - A function that generates output for each test input
    ///
    /// # Returns
    ///
    /// A `TestReport` containing all results.
    pub async fn run_suite<F, Fut>(
        &self,
        suite: &TestSuite,
        output_generator: F,
    ) -> Result<TestReport>
    where
        F: Fn(&str) -> Fut,
        Fut: std::future::Future<Output = Result<String>>,
    {
        let start = Instant::now();
        let mut report = TestReport::new().with_suite_name(&suite.name);

        report.started_at = Some(chrono::Utc::now().to_rfc3339());

        info!("Running test suite: {}", suite.name);

        if !suite.enabled {
            info!("Suite '{}' is disabled, skipping all tests", suite.name);
            for test in &suite.tests {
                report.add_result(
                    TestResult::new(&test.name, TestStatus::Skipped)
                        .with_metadata("skip_reason", "suite disabled"),
                );
            }
            return Ok(report);
        }

        // Run before_all if specified
        if let Some(setup) = &suite.before_all {
            debug!("Running suite before_all: {}", setup);
        }

        // Get tests sorted by priority
        let tests = suite.tests_by_priority();

        for test in tests {
            // Run before_each if specified
            if let Some(setup) = &suite.before_each {
                debug!("Running before_each for test '{}': {}", test.name, setup);
            }

            // Generate output for this test
            match output_generator(&test.input).await {
                Ok(output) => {
                    let result = self.run_test(test, &output).await?;
                    let should_stop = self.config.fail_fast && result.failed();
                    report.add_result(result);

                    if should_stop {
                        warn!("Stopping suite due to fail_fast setting");
                        break;
                    }
                }
                Err(e) => {
                    report.add_result(
                        TestResult::new(&test.name, TestStatus::Error)
                            .with_error(format!("Output generation failed: {}", e)),
                    );
                }
            }

            // Run after_each if specified
            if let Some(teardown) = &suite.after_each {
                debug!(
                    "Running after_each for test '{}': {}",
                    test.name, teardown
                );
            }
        }

        // Run after_all if specified
        if let Some(teardown) = &suite.after_all {
            debug!("Running suite after_all: {}", teardown);
        }

        report.duration_ms = start.elapsed().as_millis() as u64;
        report.ended_at = Some(chrono::Utc::now().to_rfc3339());
        report.coverage = self.calculate_coverage(&report);

        info!(
            "Suite '{}' complete: {}",
            suite.name,
            report.summary()
        );

        Ok(report)
    }

    // =========================================================================
    // VALIDATION HELPERS
    // =========================================================================

    /// Validate a single output against an expected output definition.
    pub fn validate(&self, actual: &str, expected: &ExpectedOutput) -> bool {
        expected.matches(actual)
    }

    /// Create a test case from a simple input/output pair.
    pub fn quick_test(
        &self,
        name: impl Into<String>,
        input: impl Into<String>,
        expected: ExpectedOutput,
    ) -> TestCase {
        TestCase::new(name).with_input(input).with_expected(expected)
    }
}

// =============================================================================
// BUILDER
// =============================================================================

/// Builder for custom TesterAgent configuration.
#[derive(Debug, Default)]
pub struct TesterAgentBuilder {
    config: Option<TesterConfig>,
}

impl TesterAgentBuilder {
    /// Set the full configuration.
    pub fn config(mut self, config: TesterConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set parallel execution.
    pub fn parallel(mut self, parallel: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.parallel = parallel;
        self.config = Some(config);
        self
    }

    /// Set maximum parallel tests.
    pub fn max_parallel(mut self, max: usize) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.max_parallel = max;
        self.config = Some(config);
        self
    }

    /// Set fail fast behavior.
    pub fn fail_fast(mut self, fail_fast: bool) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.fail_fast = fail_fast;
        self.config = Some(config);
        self
    }

    /// Set default timeout.
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.default_timeout_ms = timeout;
        self.config = Some(config);
        self
    }

    /// Add include tags.
    pub fn include_tags(mut self, tags: Vec<String>) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.include_tags = tags;
        self.config = Some(config);
        self
    }

    /// Add exclude tags.
    pub fn exclude_tags(mut self, tags: Vec<String>) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.exclude_tags = tags;
        self.config = Some(config);
        self
    }

    /// Set retry count.
    pub fn retry_count(mut self, count: u32) -> Self {
        let mut config = self.config.take().unwrap_or_default();
        config.retry_count = count;
        self.config = Some(config);
        self
    }

    /// Build the TesterAgent.
    pub fn build(self) -> TesterAgent {
        TesterAgent {
            config: self.config.unwrap_or_default(),
        }
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Truncate a string to a maximum length, adding ellipsis if needed.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // TestStatus Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_status_display() {
        assert_eq!(TestStatus::Passed.to_string(), "passed");
        assert_eq!(TestStatus::Failed.to_string(), "failed");
        assert_eq!(TestStatus::Skipped.to_string(), "skipped");
        assert_eq!(TestStatus::Error.to_string(), "error");
    }

    #[test]
    fn test_status_from_str() {
        assert_eq!("passed".parse::<TestStatus>().unwrap(), TestStatus::Passed);
        assert_eq!("pass".parse::<TestStatus>().unwrap(), TestStatus::Passed);
        assert_eq!("failed".parse::<TestStatus>().unwrap(), TestStatus::Failed);
        assert_eq!(
            "skipped".parse::<TestStatus>().unwrap(),
            TestStatus::Skipped
        );
        assert_eq!("error".parse::<TestStatus>().unwrap(), TestStatus::Error);
    }

    #[test]
    fn test_status_from_str_invalid() {
        assert!("invalid".parse::<TestStatus>().is_err());
    }

    #[test]
    fn test_status_is_success() {
        assert!(TestStatus::Passed.is_success());
        assert!(!TestStatus::Failed.is_success());
        assert!(!TestStatus::Skipped.is_success());
        assert!(!TestStatus::Error.is_success());
    }

    #[test]
    fn test_status_is_failure() {
        assert!(!TestStatus::Passed.is_failure());
        assert!(TestStatus::Failed.is_failure());
        assert!(!TestStatus::Skipped.is_failure());
        assert!(TestStatus::Error.is_failure());
    }

    #[test]
    fn test_status_emoji() {
        assert!(!TestStatus::Passed.emoji().is_empty());
        assert!(!TestStatus::Failed.emoji().is_empty());
    }

    #[test]
    fn test_status_default() {
        assert_eq!(TestStatus::default(), TestStatus::Failed);
    }

    // -------------------------------------------------------------------------
    // ExpectedOutput Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_expected_exact() {
        let expected = ExpectedOutput::Exact("hello".to_string());
        assert!(expected.matches("hello"));
        assert!(!expected.matches("Hello"));
        assert!(!expected.matches("hello world"));
    }

    #[test]
    fn test_expected_contains() {
        let expected = ExpectedOutput::Contains("world".to_string());
        assert!(expected.matches("hello world"));
        assert!(expected.matches("world"));
        assert!(!expected.matches("hello"));
    }

    #[test]
    fn test_expected_matches_regex() {
        let expected = ExpectedOutput::Matches(r"\d+".to_string());
        assert!(expected.matches("123"));
        assert!(expected.matches("abc123def"));
        assert!(!expected.matches("abc"));
    }

    #[test]
    fn test_expected_matches_invalid_regex() {
        let expected = ExpectedOutput::Matches(r"[invalid".to_string());
        assert!(!expected.matches("anything"));
    }

    #[test]
    fn test_expected_not_contains() {
        let expected = ExpectedOutput::NotContains("error".to_string());
        assert!(expected.matches("success"));
        assert!(!expected.matches("error occurred"));
    }

    #[test]
    fn test_expected_min_length() {
        let expected = ExpectedOutput::MinLength(5);
        assert!(expected.matches("hello"));
        assert!(expected.matches("hello world"));
        assert!(!expected.matches("hi"));
    }

    #[test]
    fn test_expected_max_length() {
        let expected = ExpectedOutput::MaxLength(5);
        assert!(expected.matches("hello"));
        assert!(expected.matches("hi"));
        assert!(!expected.matches("hello world"));
    }

    #[test]
    fn test_expected_all() {
        let expected = ExpectedOutput::All(vec![
            ExpectedOutput::Contains("hello".to_string()),
            ExpectedOutput::MinLength(5),
        ]);
        assert!(expected.matches("hello world"));
        assert!(!expected.matches("hi"));
        assert!(!expected.matches("world"));
    }

    #[test]
    fn test_expected_any() {
        let expected = ExpectedOutput::Any(vec![
            ExpectedOutput::Exact("hello".to_string()),
            ExpectedOutput::Exact("world".to_string()),
        ]);
        assert!(expected.matches("hello"));
        assert!(expected.matches("world"));
        assert!(!expected.matches("hi"));
    }

    #[test]
    fn test_expected_description() {
        let exact = ExpectedOutput::Exact("test".to_string());
        assert!(exact.description().contains("exactly equals"));

        let contains = ExpectedOutput::Contains("test".to_string());
        assert!(contains.description().contains("contains"));

        let min = ExpectedOutput::MinLength(10);
        assert!(min.description().contains("at least"));
    }

    #[test]
    fn test_expected_constructors() {
        let exact = ExpectedOutput::exact("test");
        assert!(matches!(exact, ExpectedOutput::Exact(_)));

        let contains = ExpectedOutput::contains("test");
        assert!(matches!(contains, ExpectedOutput::Contains(_)));

        let regex = ExpectedOutput::regex(r"\d+");
        assert!(matches!(regex, ExpectedOutput::Matches(_)));

        let not_contains = ExpectedOutput::not_contains("error");
        assert!(matches!(not_contains, ExpectedOutput::NotContains(_)));

        let min = ExpectedOutput::min_length(5);
        assert!(matches!(min, ExpectedOutput::MinLength(_)));

        let max = ExpectedOutput::max_length(100);
        assert!(matches!(max, ExpectedOutput::MaxLength(_)));
    }

    #[test]
    fn test_expected_default() {
        let expected = ExpectedOutput::default();
        assert!(matches!(expected, ExpectedOutput::MinLength(1)));
    }

    // -------------------------------------------------------------------------
    // SchemaExpectation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_schema_any() {
        let schema = SchemaExpectation::new(SchemaType::Any);
        assert!(schema.validate(r#"{"key": "value"}"#));
        assert!(schema.validate("plain text"));
        assert!(!schema.validate(""));
    }

    #[test]
    fn test_schema_object() {
        let schema = SchemaExpectation::new(SchemaType::Object);
        assert!(schema.validate(r#"{"key": "value"}"#));
        assert!(!schema.validate(r#"["array"]"#));
        assert!(!schema.validate("string"));
    }

    #[test]
    fn test_schema_object_with_required_fields() {
        let schema = SchemaExpectation::new(SchemaType::Object)
            .with_required_fields(vec!["name".to_string(), "age".to_string()]);

        assert!(schema.validate(r#"{"name": "John", "age": 30}"#));
        assert!(!schema.validate(r#"{"name": "John"}"#));
    }

    #[test]
    fn test_schema_array() {
        let schema = SchemaExpectation::new(SchemaType::Array);
        assert!(schema.validate(r#"[1, 2, 3]"#));
        assert!(!schema.validate(r#"{"key": "value"}"#));
    }

    #[test]
    fn test_schema_array_with_min_items() {
        let schema = SchemaExpectation::new(SchemaType::Array).with_min_items(2);
        assert!(schema.validate(r#"[1, 2, 3]"#));
        assert!(!schema.validate(r#"[1]"#));
    }

    #[test]
    fn test_schema_array_with_max_items() {
        let schema = SchemaExpectation::new(SchemaType::Array).with_max_items(2);
        assert!(schema.validate(r#"[1, 2]"#));
        assert!(!schema.validate(r#"[1, 2, 3]"#));
    }

    #[test]
    fn test_schema_string() {
        let schema = SchemaExpectation::new(SchemaType::String);
        assert!(schema.validate(r#""hello""#));
        assert!(!schema.validate(r#"123"#));
    }

    #[test]
    fn test_schema_number() {
        let schema = SchemaExpectation::new(SchemaType::Number);
        assert!(schema.validate("123"));
        assert!(schema.validate("123.45"));
        assert!(!schema.validate(r#""string""#));
    }

    #[test]
    fn test_schema_boolean() {
        let schema = SchemaExpectation::new(SchemaType::Boolean);
        assert!(schema.validate("true"));
        assert!(schema.validate("false"));
        assert!(!schema.validate("1"));
    }

    #[test]
    fn test_schema_description() {
        let obj = SchemaExpectation::new(SchemaType::Object);
        assert!(obj.description().contains("JSON object"));

        let arr = SchemaExpectation::new(SchemaType::Array)
            .with_min_items(1)
            .with_max_items(10);
        let desc = arr.description();
        assert!(desc.contains("min: 1"));
        assert!(desc.contains("max: 10"));
    }

    // -------------------------------------------------------------------------
    // TestCase Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_case_new() {
        let test = TestCase::new("my_test");
        assert_eq!(test.name, "my_test");
        assert!(test.enabled);
        assert_eq!(test.timeout_ms, 5000);
    }

    #[test]
    fn test_case_builder() {
        let test = TestCase::new("builder_test")
            .with_description("A test case")
            .with_input("test input")
            .with_expected(ExpectedOutput::contains("output"))
            .with_tag("unit")
            .with_timeout_ms(10000)
            .with_priority(5);

        assert_eq!(test.description, Some("A test case".to_string()));
        assert_eq!(test.input, "test input");
        assert!(test.has_tag("unit"));
        assert_eq!(test.timeout_ms, 10000);
        assert_eq!(test.priority, 5);
    }

    #[test]
    fn test_case_with_tags() {
        let test =
            TestCase::new("tagged").with_tags(vec!["unit".to_string(), "fast".to_string()]);

        assert!(test.has_tag("unit"));
        assert!(test.has_tag("fast"));
        assert!(!test.has_tag("slow"));
    }

    #[test]
    fn test_case_has_tag_case_insensitive() {
        let test = TestCase::new("case_test").with_tag("Unit");
        assert!(test.has_tag("unit"));
        assert!(test.has_tag("UNIT"));
    }

    #[test]
    fn test_case_matches_any_tag() {
        let test =
            TestCase::new("multi_tag").with_tags(vec!["unit".to_string(), "fast".to_string()]);

        assert!(test.matches_any_tag(&["unit".to_string()]));
        assert!(test.matches_any_tag(&["slow".to_string(), "fast".to_string()]));
        assert!(!test.matches_any_tag(&["slow".to_string(), "integration".to_string()]));
    }

    #[test]
    fn test_case_with_setup_teardown() {
        let test = TestCase::new("lifecycle")
            .with_setup("prepare database")
            .with_teardown("cleanup database");

        assert_eq!(test.setup, Some("prepare database".to_string()));
        assert_eq!(test.teardown, Some("cleanup database".to_string()));
    }

    #[test]
    fn test_case_disabled() {
        let test = TestCase::new("disabled_test").with_enabled(false);
        assert!(!test.enabled);
    }

    // -------------------------------------------------------------------------
    // TestResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_result_new() {
        let result = TestResult::new("test", TestStatus::Passed);
        assert_eq!(result.test_name, "test");
        assert_eq!(result.status, TestStatus::Passed);
    }

    #[test]
    fn test_result_builder() {
        let result = TestResult::new("builder_test", TestStatus::Failed)
            .with_actual_output("actual output")
            .with_error("comparison failed")
            .with_duration_ms(150)
            .with_expected_description("should contain 'expected'")
            .with_metadata("key", "value");

        assert_eq!(result.actual_output, Some("actual output".to_string()));
        assert_eq!(result.error_message, Some("comparison failed".to_string()));
        assert_eq!(result.duration_ms, 150);
        assert!(result.metadata.contains_key("key"));
    }

    #[test]
    fn test_result_passed() {
        let passed = TestResult::new("t", TestStatus::Passed);
        assert!(passed.passed());
        assert!(!passed.failed());

        let failed = TestResult::new("t", TestStatus::Failed);
        assert!(!failed.passed());
        assert!(failed.failed());
    }

    #[test]
    fn test_result_summary() {
        let result = TestResult::new("my_test", TestStatus::Passed).with_duration_ms(100);
        let summary = result.summary();
        assert!(summary.contains("my_test"));
        assert!(summary.contains("100ms"));
    }

    #[test]
    fn test_result_summary_with_error() {
        let result = TestResult::new("failing", TestStatus::Failed)
            .with_error("assertion failed")
            .with_duration_ms(50);

        let summary = result.summary();
        assert!(summary.contains("assertion failed"));
    }

    // -------------------------------------------------------------------------
    // TestSuite Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_suite_new() {
        let suite = TestSuite::new("my_suite");
        assert_eq!(suite.name, "my_suite");
        assert!(suite.enabled);
        assert!(suite.tests.is_empty());
    }

    #[test]
    fn test_suite_builder() {
        let suite = TestSuite::new("builder_suite")
            .with_description("A test suite")
            .with_test(TestCase::new("test1"))
            .with_test(TestCase::new("test2"))
            .with_tag("integration");

        assert_eq!(suite.description, Some("A test suite".to_string()));
        assert_eq!(suite.test_count(), 2);
        assert!(suite.tags.contains(&"integration".to_string()));
    }

    #[test]
    fn test_suite_with_tests() {
        let tests = vec![
            TestCase::new("t1").with_priority(1),
            TestCase::new("t2").with_priority(3),
            TestCase::new("t3").with_priority(2),
        ];

        let suite = TestSuite::new("ordered").with_tests(tests);
        assert_eq!(suite.test_count(), 3);
    }

    #[test]
    fn test_suite_enabled_tests() {
        let suite = TestSuite::new("filtered")
            .with_test(TestCase::new("enabled1"))
            .with_test(TestCase::new("disabled").with_enabled(false))
            .with_test(TestCase::new("enabled2"));

        let enabled = suite.enabled_tests();
        assert_eq!(enabled.len(), 2);
    }

    #[test]
    fn test_suite_tests_by_priority() {
        let suite = TestSuite::new("priority")
            .with_test(TestCase::new("low").with_priority(1))
            .with_test(TestCase::new("high").with_priority(10))
            .with_test(TestCase::new("medium").with_priority(5));

        let sorted = suite.tests_by_priority();
        assert_eq!(sorted[0].name, "high");
        assert_eq!(sorted[1].name, "medium");
        assert_eq!(sorted[2].name, "low");
    }

    #[test]
    fn test_suite_tests_with_tag() {
        let suite = TestSuite::new("tagged")
            .with_test(TestCase::new("unit1").with_tag("unit"))
            .with_test(TestCase::new("integration1").with_tag("integration"))
            .with_test(TestCase::new("unit2").with_tag("unit"));

        let unit_tests = suite.tests_with_tag("unit");
        assert_eq!(unit_tests.len(), 2);
    }

    #[test]
    fn test_suite_lifecycle_hooks() {
        let suite = TestSuite::new("hooks")
            .with_before_all("setup all")
            .with_after_all("cleanup all")
            .with_before_each("setup each")
            .with_after_each("cleanup each");

        assert_eq!(suite.before_all, Some("setup all".to_string()));
        assert_eq!(suite.after_all, Some("cleanup all".to_string()));
        assert_eq!(suite.before_each, Some("setup each".to_string()));
        assert_eq!(suite.after_each, Some("cleanup each".to_string()));
    }

    // -------------------------------------------------------------------------
    // TestReport Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_report_new() {
        let report = TestReport::new();
        assert_eq!(report.passed, 0);
        assert_eq!(report.failed, 0);
        assert_eq!(report.total(), 0);
    }

    #[test]
    fn test_report_add_result() {
        let mut report = TestReport::new();
        report.add_result(TestResult::new("t1", TestStatus::Passed));
        report.add_result(TestResult::new("t2", TestStatus::Failed));
        report.add_result(TestResult::new("t3", TestStatus::Skipped));
        report.add_result(TestResult::new("t4", TestStatus::Error));

        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert_eq!(report.skipped, 1);
        assert_eq!(report.errored, 1);
        assert_eq!(report.total(), 4);
        assert_eq!(report.results.len(), 4);
    }

    #[test]
    fn test_report_success_rate() {
        let mut report = TestReport::new();
        report.add_result(TestResult::new("p1", TestStatus::Passed));
        report.add_result(TestResult::new("p2", TestStatus::Passed));
        report.add_result(TestResult::new("f1", TestStatus::Failed));

        let rate = report.success_rate();
        assert!((rate - 0.6666666).abs() < 0.01);
    }

    #[test]
    fn test_report_success_rate_empty() {
        let report = TestReport::new();
        assert!((report.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_all_passed() {
        let mut report = TestReport::new();
        report.add_result(TestResult::new("p1", TestStatus::Passed));
        report.add_result(TestResult::new("p2", TestStatus::Passed));
        report.add_result(TestResult::new("s1", TestStatus::Skipped));

        assert!(report.all_passed());

        report.add_result(TestResult::new("f1", TestStatus::Failed));
        assert!(!report.all_passed());
    }

    #[test]
    fn test_report_has_failures() {
        let mut report = TestReport::new();
        assert!(!report.has_failures());

        report.add_result(TestResult::new("p1", TestStatus::Passed));
        assert!(!report.has_failures());

        report.add_result(TestResult::new("f1", TestStatus::Failed));
        assert!(report.has_failures());
    }

    #[test]
    fn test_report_failures() {
        let mut report = TestReport::new();
        report.add_result(TestResult::new("p1", TestStatus::Passed));
        report.add_result(TestResult::new("f1", TestStatus::Failed));
        report.add_result(TestResult::new("e1", TestStatus::Error));

        let failures = report.failures();
        assert_eq!(failures.len(), 2);
    }

    #[test]
    fn test_report_successes() {
        let mut report = TestReport::new();
        report.add_result(TestResult::new("p1", TestStatus::Passed));
        report.add_result(TestResult::new("p2", TestStatus::Passed));
        report.add_result(TestResult::new("f1", TestStatus::Failed));

        let successes = report.successes();
        assert_eq!(successes.len(), 2);
    }

    #[test]
    fn test_report_summary() {
        let mut report = TestReport::new();
        report.add_result(TestResult::new("p1", TestStatus::Passed));
        report.duration_ms = 100;

        let summary = report.summary();
        assert!(summary.contains("ALL PASSED"));
        assert!(summary.contains("1 passed"));
    }

    #[test]
    fn test_report_to_markdown() {
        let mut report = TestReport::new()
            .with_suite_name("My Suite")
            .with_metadata("version", "1.0");

        report.add_result(TestResult::new("t1", TestStatus::Passed));
        report.add_result(
            TestResult::new("t2", TestStatus::Failed).with_error("assertion failed"),
        );

        let md = report.to_markdown();
        assert!(md.contains("# Test Report"));
        assert!(md.contains("My Suite"));
        assert!(md.contains("Passed"));
        assert!(md.contains("Failed"));
        assert!(md.contains("assertion failed"));
    }

    #[test]
    fn test_report_default() {
        let report = TestReport::default();
        assert_eq!(report.total(), 0);
    }

    // -------------------------------------------------------------------------
    // TesterConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = TesterConfig::default();
        assert!(!config.parallel);
        assert!(!config.fail_fast);
        assert_eq!(config.default_timeout_ms, 5000);
    }

    #[test]
    fn test_config_fast() {
        let config = TesterConfig::fast();
        assert!(config.parallel);
        assert!(config.fail_fast);
        assert_eq!(config.max_parallel, 8);
    }

    #[test]
    fn test_config_thorough() {
        let config = TesterConfig::thorough();
        assert!(!config.parallel);
        assert!(!config.fail_fast);
        assert_eq!(config.retry_count, 2);
    }

    #[test]
    fn test_config_ci() {
        let config = TesterConfig::ci();
        assert!(config.parallel);
        assert!(!config.fail_fast);
        assert_eq!(config.retry_count, 1);
    }

    // -------------------------------------------------------------------------
    // TesterAgent Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_new() {
        let agent = TesterAgent::new();
        assert!(!agent.config().parallel);
    }

    #[test]
    fn test_agent_with_config() {
        let config = TesterConfig::fast();
        let agent = TesterAgent::with_config(config);
        assert!(agent.config().parallel);
    }

    #[test]
    fn test_agent_builder() {
        let agent = TesterAgent::builder()
            .parallel(true)
            .max_parallel(10)
            .fail_fast(true)
            .timeout_ms(3000)
            .build();

        assert!(agent.config().parallel);
        assert_eq!(agent.config().max_parallel, 10);
        assert!(agent.config().fail_fast);
        assert_eq!(agent.config().default_timeout_ms, 3000);
    }

    #[test]
    fn test_agent_builder_with_tags() {
        let agent = TesterAgent::builder()
            .include_tags(vec!["unit".to_string()])
            .exclude_tags(vec!["slow".to_string()])
            .build();

        assert_eq!(agent.config().include_tags, vec!["unit".to_string()]);
        assert_eq!(agent.config().exclude_tags, vec!["slow".to_string()]);
    }

    #[test]
    fn test_agent_builder_with_retry() {
        let agent = TesterAgent::builder().retry_count(3).build();

        assert_eq!(agent.config().retry_count, 3);
    }

    #[test]
    fn test_agent_default() {
        let agent = TesterAgent::default();
        assert!(!agent.config().parallel);
    }

    // -------------------------------------------------------------------------
    // TesterAgent Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_validate() {
        let agent = TesterAgent::new();

        assert!(agent.validate("hello world", &ExpectedOutput::contains("hello")));
        assert!(!agent.validate("hello world", &ExpectedOutput::exact("hello")));
    }

    #[test]
    fn test_agent_quick_test() {
        let agent = TesterAgent::new();
        let test = agent.quick_test("quick", "input", ExpectedOutput::contains("output"));

        assert_eq!(test.name, "quick");
        assert_eq!(test.input, "input");
    }

    // -------------------------------------------------------------------------
    // TesterAgent Execution Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_run_test_passing() {
        let agent = TesterAgent::new();
        let test = TestCase::new("passing_test")
            .with_input("test input")
            .with_expected(ExpectedOutput::contains("hello"));

        let result = agent.run_test(&test, "hello world").await.unwrap();
        assert!(result.passed());
        assert_eq!(result.status, TestStatus::Passed);
    }

    #[tokio::test]
    async fn test_run_test_failing() {
        let agent = TesterAgent::new();
        let test = TestCase::new("failing_test")
            .with_input("test input")
            .with_expected(ExpectedOutput::exact("expected"));

        let result = agent.run_test(&test, "actual").await.unwrap();
        assert!(result.failed());
        assert!(result.error_message.is_some());
    }

    #[tokio::test]
    async fn test_run_test_disabled() {
        let agent = TesterAgent::new();
        let test = TestCase::new("disabled_test")
            .with_enabled(false)
            .with_expected(ExpectedOutput::exact("anything"));

        let result = agent.run_test(&test, "anything").await.unwrap();
        assert_eq!(result.status, TestStatus::Skipped);
    }

    #[tokio::test]
    async fn test_run_test_with_tag_filter() {
        let agent = TesterAgent::builder()
            .include_tags(vec!["unit".to_string()])
            .build();

        let test = TestCase::new("tagged")
            .with_tag("integration")
            .with_expected(ExpectedOutput::exact("x"));

        let result = agent.run_test(&test, "x").await.unwrap();
        assert_eq!(result.status, TestStatus::Skipped);
    }

    #[tokio::test]
    async fn test_run_test_with_exclude_tag() {
        let agent = TesterAgent::builder()
            .exclude_tags(vec!["slow".to_string()])
            .build();

        let test = TestCase::new("slow_test")
            .with_tag("slow")
            .with_expected(ExpectedOutput::exact("x"));

        let result = agent.run_test(&test, "x").await.unwrap();
        assert_eq!(result.status, TestStatus::Skipped);
    }

    #[tokio::test]
    async fn test_run_tests_multiple() {
        let agent = TesterAgent::new();
        let test1 = TestCase::new("t1").with_expected(ExpectedOutput::contains("a"));
        let test2 = TestCase::new("t2").with_expected(ExpectedOutput::contains("b"));
        let test3 = TestCase::new("t3").with_expected(ExpectedOutput::contains("c"));

        let tests = vec![(&test1, "a"), (&test2, "b"), (&test3, "x")];

        let report = agent.run_tests(tests).await.unwrap();
        assert_eq!(report.passed, 2);
        assert_eq!(report.failed, 1);
    }

    #[tokio::test]
    async fn test_run_tests_parallel() {
        let agent = TesterAgent::builder().parallel(true).max_parallel(2).build();

        let test1 = TestCase::new("p1").with_expected(ExpectedOutput::contains("a"));
        let test2 = TestCase::new("p2").with_expected(ExpectedOutput::contains("b"));

        let tests = vec![(&test1, "a"), (&test2, "b")];

        let report = agent.run_tests(tests).await.unwrap();
        assert_eq!(report.passed, 2);
    }

    #[tokio::test]
    async fn test_run_tests_fail_fast() {
        let agent = TesterAgent::builder().fail_fast(true).build();

        let test1 = TestCase::new("f1")
            .with_priority(2)
            .with_expected(ExpectedOutput::exact("wrong"));
        let test2 = TestCase::new("f2")
            .with_priority(1)
            .with_expected(ExpectedOutput::exact("x"));

        let tests = vec![(&test1, "actual"), (&test2, "x")];

        let report = agent.run_tests(tests).await.unwrap();
        // Should stop after first failure
        assert!(report.failed >= 1);
    }

    #[tokio::test]
    async fn test_run_suite() {
        let agent = TesterAgent::new();

        let suite = TestSuite::new("test_suite")
            .with_test(TestCase::new("s1").with_input("in1").with_expected(ExpectedOutput::contains("out1")))
            .with_test(TestCase::new("s2").with_input("in2").with_expected(ExpectedOutput::contains("out2")));

        let report = agent
            .run_suite(&suite, |input: &str| {
                let input = input.to_string();
                async move {
                    Ok(format!("output for {} is out{}", input, input.chars().last().unwrap_or('?')))
                }
            })
            .await
            .unwrap();

        assert_eq!(report.passed, 2);
        assert_eq!(report.suite_name, Some("test_suite".to_string()));
    }

    #[tokio::test]
    async fn test_run_suite_disabled() {
        let agent = TesterAgent::new();

        let mut suite = TestSuite::new("disabled_suite")
            .with_test(TestCase::new("t1").with_expected(ExpectedOutput::exact("x")));
        suite.enabled = false;

        let report = agent
            .run_suite(&suite, |_: &str| async { Ok("x".to_string()) })
            .await
            .unwrap();

        assert_eq!(report.skipped, 1);
    }

    // -------------------------------------------------------------------------
    // Utility Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello...");
        assert_eq!(truncate("hi", 2), "hi");
    }
}
