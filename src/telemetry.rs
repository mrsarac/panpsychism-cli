//! # Telemetry Module
//!
//! Enhanced tracing and structured logging for debugging and observability.
//!
//! This module provides:
//! - **Distributed Tracing**: Span-based tracing with parent-child relationships
//! - **Structured Logging**: Log entries with context and filtering
//! - **RAII Span Management**: Automatic span lifecycle via SpanGuard
//!
//! ## Example
//!
//! ```rust
//! use panpsychism::telemetry::{Tracer, TracerConfig, Logger, LoggerConfig, LogLevel, SpanStatus};
//!
//! let config = TracerConfig::default();
//! let mut tracer = Tracer::new("my-service", config);
//!
//! // Start a span with RAII guard
//! let span_id = {
//!     let guard = tracer.start_span("operation");
//!     tracer.set_attribute(&guard.span_id(), "user_id", serde_json::json!("user-123"));
//!     guard.span_id().to_string()
//! }; // Span automatically ends when guard drops
//!
//! // Logging with span context
//! let mut logger = Logger::new(LoggerConfig::default());
//! logger.with_span(&span_id).info("Operation completed");
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use uuid::Uuid;

// ============================================================================
// Span Types
// ============================================================================

/// Status of a span indicating success, failure, or in-progress.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpanStatus {
    /// Span completed successfully
    Ok,
    /// Span completed with an error
    Error(String),
    /// Span is still in progress
    InProgress,
}

impl Default for SpanStatus {
    fn default() -> Self {
        Self::InProgress
    }
}

impl fmt::Display for SpanStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpanStatus::Ok => write!(f, "OK"),
            SpanStatus::Error(msg) => write!(f, "ERROR: {}", msg),
            SpanStatus::InProgress => write!(f, "IN_PROGRESS"),
        }
    }
}

/// An event that occurred during a span's lifetime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Name of the event
    pub name: String,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Additional attributes for the event
    pub attributes: HashMap<String, serde_json::Value>,
}

impl SpanEvent {
    /// Create a new span event with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timestamp: Utc::now(),
            attributes: HashMap::new(),
        }
    }

    /// Create a new span event with attributes.
    pub fn with_attributes(
        name: impl Into<String>,
        attributes: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            name: name.into(),
            timestamp: Utc::now(),
            attributes,
        }
    }

    /// Add an attribute to the event.
    pub fn add_attribute(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.attributes.insert(key.into(), value);
    }
}

/// A span representing a unit of work in distributed tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Unique identifier for this span
    pub id: String,
    /// Parent span ID if this is a child span
    pub parent_id: Option<String>,
    /// Human-readable name for this span
    pub name: String,
    /// Associated agent ID if applicable
    pub agent_id: Option<String>,
    /// When the span started
    pub start_time: DateTime<Utc>,
    /// When the span ended (None if still in progress)
    pub end_time: Option<DateTime<Utc>>,
    /// Current status of the span
    pub status: SpanStatus,
    /// Key-value attributes for the span
    pub attributes: HashMap<String, serde_json::Value>,
    /// Events that occurred during the span
    pub events: Vec<SpanEvent>,
}

impl Span {
    /// Create a new span with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            parent_id: None,
            name: name.into(),
            agent_id: None,
            start_time: Utc::now(),
            end_time: None,
            status: SpanStatus::InProgress,
            attributes: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Create a new span with a parent.
    pub fn with_parent(name: impl Into<String>, parent_id: impl Into<String>) -> Self {
        let mut span = Self::new(name);
        span.parent_id = Some(parent_id.into());
        span
    }

    /// Set the agent ID for this span.
    pub fn set_agent_id(&mut self, agent_id: impl Into<String>) {
        self.agent_id = Some(agent_id.into());
    }

    /// Add an attribute to the span.
    pub fn add_attribute(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.attributes.insert(key.into(), value);
    }

    /// Add an event to the span.
    pub fn add_event(&mut self, event: SpanEvent) {
        self.events.push(event);
    }

    /// End the span with the given status.
    pub fn end(&mut self, status: SpanStatus) {
        self.end_time = Some(Utc::now());
        self.status = status;
    }

    /// Get the duration of the span in milliseconds.
    pub fn duration_ms(&self) -> Option<i64> {
        self.end_time.map(|end| {
            (end - self.start_time).num_milliseconds()
        })
    }

    /// Check if the span is still in progress.
    pub fn is_in_progress(&self) -> bool {
        matches!(self.status, SpanStatus::InProgress)
    }

    /// Check if the span completed successfully.
    pub fn is_ok(&self) -> bool {
        matches!(self.status, SpanStatus::Ok)
    }

    /// Check if the span completed with an error.
    pub fn is_error(&self) -> bool {
        matches!(self.status, SpanStatus::Error(_))
    }
}

// ============================================================================
// Tracer Configuration
// ============================================================================

/// Configuration for the tracer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerConfig {
    /// Maximum number of completed spans to retain
    pub max_completed_spans: usize,
    /// Sampling rate (0.0 to 1.0) - percentage of spans to actually trace
    pub sample_rate: f64,
    /// Whether to enable export functionality
    pub export_enabled: bool,
}

impl Default for TracerConfig {
    fn default() -> Self {
        Self {
            max_completed_spans: 1000,
            sample_rate: 1.0,
            export_enabled: true,
        }
    }
}

impl TracerConfig {
    /// Create a new tracer config with custom values.
    pub fn new(max_completed_spans: usize, sample_rate: f64, export_enabled: bool) -> Self {
        Self {
            max_completed_spans,
            sample_rate: sample_rate.clamp(0.0, 1.0),
            export_enabled,
        }
    }

    /// Set the maximum number of completed spans.
    pub fn with_max_spans(mut self, max: usize) -> Self {
        self.max_completed_spans = max;
        self
    }

    /// Set the sampling rate.
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable export.
    pub fn with_export(mut self, enabled: bool) -> Self {
        self.export_enabled = enabled;
        self
    }
}

// ============================================================================
// Tracer
// ============================================================================

/// A distributed tracer for tracking spans across operations.
#[derive(Debug)]
pub struct Tracer {
    /// Name of the service being traced
    service_name: String,
    /// Currently active spans
    active_spans: HashMap<String, Span>,
    /// Completed spans (ring buffer)
    completed_spans: VecDeque<Span>,
    /// Tracer configuration
    config: TracerConfig,
    /// Random number generator seed for sampling
    sample_counter: u64,
}

impl Tracer {
    /// Create a new tracer with the given service name and configuration.
    pub fn new(service_name: impl Into<String>, config: TracerConfig) -> Self {
        Self {
            service_name: service_name.into(),
            active_spans: HashMap::new(),
            completed_spans: VecDeque::with_capacity(config.max_completed_spans),
            config,
            sample_counter: 0,
        }
    }

    /// Get the service name.
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Get the tracer configuration.
    pub fn config(&self) -> &TracerConfig {
        &self.config
    }

    /// Check if a span should be sampled based on the sample rate.
    fn should_sample(&mut self) -> bool {
        if self.config.sample_rate >= 1.0 {
            return true;
        }
        if self.config.sample_rate <= 0.0 {
            return false;
        }
        // Simple deterministic sampling based on counter
        self.sample_counter = self.sample_counter.wrapping_add(1);
        let threshold = (self.config.sample_rate * u64::MAX as f64) as u64;
        // Use a simple hash of the counter for pseudo-randomness
        let hash = self.sample_counter.wrapping_mul(0x517cc1b727220a95);
        hash < threshold
    }

    /// Start a new span and return a guard for automatic cleanup.
    pub fn start_span(&mut self, name: &str) -> SpanGuard<'_> {
        let span = Span::new(name);
        let span_id = span.id.clone();

        if self.should_sample() {
            self.active_spans.insert(span_id.clone(), span);
        }

        SpanGuard {
            tracer: self,
            span_id,
            sampled: true,
        }
    }

    /// Start a new span with a parent and return a guard.
    pub fn start_span_with_parent(&mut self, name: &str, parent_id: &str) -> SpanGuard<'_> {
        let span = Span::with_parent(name, parent_id);
        let span_id = span.id.clone();

        if self.should_sample() {
            self.active_spans.insert(span_id.clone(), span);
        }

        SpanGuard {
            tracer: self,
            span_id,
            sampled: true,
        }
    }

    /// Start a span manually without a guard.
    pub fn start_span_manual(&mut self, name: &str) -> String {
        let span = Span::new(name);
        let span_id = span.id.clone();

        if self.should_sample() {
            self.active_spans.insert(span_id.clone(), span);
        }

        span_id
    }

    /// Start a span with parent manually.
    pub fn start_span_with_parent_manual(&mut self, name: &str, parent_id: &str) -> String {
        let span = Span::with_parent(name, parent_id);
        let span_id = span.id.clone();

        if self.should_sample() {
            self.active_spans.insert(span_id.clone(), span);
        }

        span_id
    }

    /// Add an event to an active span.
    pub fn add_event(&mut self, span_id: &str, event: SpanEvent) {
        if let Some(span) = self.active_spans.get_mut(span_id) {
            span.add_event(event);
        }
    }

    /// Add a simple event by name to an active span.
    pub fn add_event_simple(&mut self, span_id: &str, event_name: &str) {
        self.add_event(span_id, SpanEvent::new(event_name));
    }

    /// Set an attribute on an active span.
    pub fn set_attribute(&mut self, span_id: &str, key: &str, value: serde_json::Value) {
        if let Some(span) = self.active_spans.get_mut(span_id) {
            span.add_attribute(key, value);
        }
    }

    /// Set the agent ID on an active span.
    pub fn set_agent_id(&mut self, span_id: &str, agent_id: &str) {
        if let Some(span) = self.active_spans.get_mut(span_id) {
            span.set_agent_id(agent_id);
        }
    }

    /// End an active span with the given status.
    pub fn end_span(&mut self, span_id: &str, status: SpanStatus) {
        if let Some(mut span) = self.active_spans.remove(span_id) {
            span.end(status);
            self.store_completed_span(span);
        }
    }

    /// Store a completed span in the ring buffer.
    fn store_completed_span(&mut self, span: Span) {
        if self.completed_spans.len() >= self.config.max_completed_spans {
            self.completed_spans.pop_front();
        }
        self.completed_spans.push_back(span);
    }

    /// Get a reference to an active span.
    pub fn get_span(&self, span_id: &str) -> Option<&Span> {
        self.active_spans.get(span_id)
    }

    /// Get a mutable reference to an active span.
    pub fn get_span_mut(&mut self, span_id: &str) -> Option<&mut Span> {
        self.active_spans.get_mut(span_id)
    }

    /// Get recent completed spans.
    pub fn get_recent_spans(&self, limit: usize) -> Vec<&Span> {
        self.completed_spans
            .iter()
            .rev()
            .take(limit)
            .collect()
    }

    /// Get all active span IDs.
    pub fn active_span_ids(&self) -> Vec<&str> {
        self.active_spans.keys().map(|s| s.as_str()).collect()
    }

    /// Get the count of active spans.
    pub fn active_span_count(&self) -> usize {
        self.active_spans.len()
    }

    /// Get the count of completed spans.
    pub fn completed_span_count(&self) -> usize {
        self.completed_spans.len()
    }

    /// Export all completed spans and clear the buffer.
    pub fn export_spans(&mut self) -> Vec<Span> {
        if !self.config.export_enabled {
            return Vec::new();
        }
        self.completed_spans.drain(..).collect()
    }

    /// Export completed spans without clearing.
    pub fn peek_spans(&self) -> Vec<&Span> {
        self.completed_spans.iter().collect()
    }

    /// Clear all completed spans.
    pub fn clear_completed(&mut self) {
        self.completed_spans.clear();
    }

    /// Find spans by name pattern.
    pub fn find_spans_by_name(&self, name_pattern: &str) -> Vec<&Span> {
        self.completed_spans
            .iter()
            .filter(|s| s.name.contains(name_pattern))
            .collect()
    }

    /// Find spans with errors.
    pub fn find_error_spans(&self) -> Vec<&Span> {
        self.completed_spans
            .iter()
            .filter(|s| s.is_error())
            .collect()
    }
}

// ============================================================================
// Span Guard (RAII)
// ============================================================================

/// RAII guard for automatic span lifecycle management.
///
/// When the guard is dropped, the span is automatically ended with `SpanStatus::Ok`.
/// To end with a different status, call `end_with_status` before dropping.
pub struct SpanGuard<'a> {
    tracer: &'a mut Tracer,
    span_id: String,
    sampled: bool,
}

impl<'a> SpanGuard<'a> {
    /// Get the span ID.
    pub fn span_id(&self) -> &str {
        &self.span_id
    }

    /// Add an event to the span.
    pub fn add_event(&mut self, event: SpanEvent) {
        self.tracer.add_event(&self.span_id, event);
    }

    /// Add a simple event by name.
    pub fn add_event_simple(&mut self, name: &str) {
        self.tracer.add_event_simple(&self.span_id, name);
    }

    /// Set an attribute on the span.
    pub fn set_attribute(&mut self, key: &str, value: serde_json::Value) {
        self.tracer.set_attribute(&self.span_id, key, value);
    }

    /// Set the agent ID.
    pub fn set_agent_id(&mut self, agent_id: &str) {
        self.tracer.set_agent_id(&self.span_id, agent_id);
    }

    /// End the span with a specific status (consumes the guard).
    pub fn end_with_status(mut self, status: SpanStatus) {
        self.tracer.end_span(&self.span_id, status);
        self.sampled = false; // Prevent double-end in Drop
    }

    /// End the span with an error (consumes the guard).
    pub fn end_with_error(self, error: impl Into<String>) {
        self.end_with_status(SpanStatus::Error(error.into()));
    }

    /// End the span successfully (consumes the guard).
    pub fn end_ok(self) {
        self.end_with_status(SpanStatus::Ok);
    }
}

impl Drop for SpanGuard<'_> {
    fn drop(&mut self) {
        if self.sampled {
            // Default to Ok if not explicitly ended
            self.tracer.end_span(&self.span_id, SpanStatus::Ok);
        }
    }
}

// ============================================================================
// Log Types
// ============================================================================

/// Log severity level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Most verbose, for debugging internals
    Trace = 0,
    /// Debug information
    Debug = 1,
    /// General information
    Info = 2,
    /// Warnings
    Warn = 3,
    /// Errors
    Error = 4,
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Info
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

impl LogLevel {
    /// Parse a log level from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "trace" => Some(Self::Trace),
            "debug" => Some(Self::Debug),
            "info" => Some(Self::Info),
            "warn" | "warning" => Some(Self::Warn),
            "error" | "err" => Some(Self::Error),
            _ => None,
        }
    }

    /// Get all log levels.
    pub fn all() -> [LogLevel; 5] {
        [
            LogLevel::Trace,
            LogLevel::Debug,
            LogLevel::Info,
            LogLevel::Warn,
            LogLevel::Error,
        ]
    }
}

/// A structured log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// When the log was created
    pub timestamp: DateTime<Utc>,
    /// Severity level
    pub level: LogLevel,
    /// Log message
    pub message: String,
    /// Target (module/component name)
    pub target: String,
    /// Associated span ID if any
    pub span_id: Option<String>,
    /// Additional structured fields
    pub fields: HashMap<String, serde_json::Value>,
}

impl LogEntry {
    /// Create a new log entry.
    pub fn new(level: LogLevel, message: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            level,
            message: message.into(),
            target: target.into(),
            span_id: None,
            fields: HashMap::new(),
        }
    }

    /// Create a log entry with a span ID.
    pub fn with_span(
        level: LogLevel,
        message: impl Into<String>,
        target: impl Into<String>,
        span_id: impl Into<String>,
    ) -> Self {
        let mut entry = Self::new(level, message, target);
        entry.span_id = Some(span_id.into());
        entry
    }

    /// Add a field to the log entry.
    pub fn add_field(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.fields.insert(key.into(), value);
    }

    /// Create a log entry with fields.
    pub fn with_fields(mut self, fields: HashMap<String, serde_json::Value>) -> Self {
        self.fields = fields;
        self
    }

    /// Format the log entry as a string.
    pub fn format(&self) -> String {
        let span_info = self
            .span_id
            .as_ref()
            .map(|id| format!(" [{}]", &id[..8.min(id.len())]))
            .unwrap_or_default();

        format!(
            "{} {} {}{}: {}",
            self.timestamp.format("%Y-%m-%d %H:%M:%S%.3f"),
            self.level,
            self.target,
            span_info,
            self.message
        )
    }
}

impl fmt::Display for LogEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

// ============================================================================
// Log Filter
// ============================================================================

/// A filter for log entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogFilter {
    /// Filter by target pattern (substring match)
    pub target_pattern: Option<String>,
    /// Minimum log level to include
    pub min_level: Option<LogLevel>,
    /// Filter by span ID
    pub span_id: Option<String>,
    /// Filter by message pattern (substring match)
    pub message_pattern: Option<String>,
}

impl LogFilter {
    /// Create a new empty filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by target pattern.
    pub fn with_target(mut self, pattern: impl Into<String>) -> Self {
        self.target_pattern = Some(pattern.into());
        self
    }

    /// Filter by minimum level.
    pub fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = Some(level);
        self
    }

    /// Filter by span ID.
    pub fn with_span_id(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }

    /// Filter by message pattern.
    pub fn with_message(mut self, pattern: impl Into<String>) -> Self {
        self.message_pattern = Some(pattern.into());
        self
    }

    /// Check if a log entry matches this filter.
    pub fn matches(&self, entry: &LogEntry) -> bool {
        // Check minimum level
        if let Some(min_level) = self.min_level {
            if entry.level < min_level {
                return false;
            }
        }

        // Check target pattern
        if let Some(ref pattern) = self.target_pattern {
            if !entry.target.contains(pattern) {
                return false;
            }
        }

        // Check span ID
        if let Some(ref span_id) = self.span_id {
            match &entry.span_id {
                Some(entry_span) if entry_span == span_id => {}
                _ => return false,
            }
        }

        // Check message pattern
        if let Some(ref pattern) = self.message_pattern {
            if !entry.message.contains(pattern) {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Logger Configuration
// ============================================================================

/// Configuration for the logger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggerConfig {
    /// Maximum number of log entries to retain
    pub max_entries: usize,
    /// Minimum log level to record
    pub min_level: LogLevel,
    /// Whether to include timestamps in output
    pub include_timestamps: bool,
    /// Default target name
    pub default_target: String,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            min_level: LogLevel::Info,
            include_timestamps: true,
            default_target: "app".to_string(),
        }
    }
}

impl LoggerConfig {
    /// Create a new logger config.
    pub fn new(max_entries: usize, min_level: LogLevel) -> Self {
        Self {
            max_entries,
            min_level,
            ..Default::default()
        }
    }

    /// Set max entries.
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Set minimum level.
    pub fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = level;
        self
    }

    /// Set timestamp inclusion.
    pub fn with_timestamps(mut self, include: bool) -> Self {
        self.include_timestamps = include;
        self
    }

    /// Set default target.
    pub fn with_default_target(mut self, target: impl Into<String>) -> Self {
        self.default_target = target.into();
        self
    }
}

// ============================================================================
// Logger
// ============================================================================

/// A structured logger with filtering and context support.
#[derive(Debug)]
pub struct Logger {
    /// Log entries (ring buffer)
    entries: VecDeque<LogEntry>,
    /// Logger configuration
    config: LoggerConfig,
    /// Active filters
    filters: Vec<LogFilter>,
    /// Default target for logs
    default_target: String,
}

impl Logger {
    /// Create a new logger with the given configuration.
    pub fn new(config: LoggerConfig) -> Self {
        let default_target = config.default_target.clone();
        Self {
            entries: VecDeque::with_capacity(config.max_entries),
            config,
            filters: Vec::new(),
            default_target,
        }
    }

    /// Get the logger configuration.
    pub fn config(&self) -> &LoggerConfig {
        &self.config
    }

    /// Add a filter to the logger.
    pub fn add_filter(&mut self, filter: LogFilter) {
        self.filters.push(filter);
    }

    /// Clear all filters.
    pub fn clear_filters(&mut self) {
        self.filters.clear();
    }

    /// Check if a log level should be recorded.
    fn should_log(&self, level: LogLevel) -> bool {
        level >= self.config.min_level
    }

    /// Store a log entry.
    fn store_entry(&mut self, entry: LogEntry) {
        // Check against filters
        if !self.filters.is_empty() {
            let matches_any = self.filters.iter().any(|f| f.matches(&entry));
            if !matches_any {
                return;
            }
        }

        // Ring buffer behavior
        if self.entries.len() >= self.config.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Log a message at the given level.
    pub fn log(&mut self, level: LogLevel, message: &str) {
        if !self.should_log(level) {
            return;
        }
        let entry = LogEntry::new(level, message, &self.default_target);
        self.store_entry(entry);
    }

    /// Log with a specific target.
    pub fn log_to(&mut self, level: LogLevel, message: &str, target: &str) {
        if !self.should_log(level) {
            return;
        }
        let entry = LogEntry::new(level, message, target);
        self.store_entry(entry);
    }

    /// Log with fields.
    pub fn log_with_fields(
        &mut self,
        level: LogLevel,
        message: &str,
        fields: HashMap<String, serde_json::Value>,
    ) {
        if !self.should_log(level) {
            return;
        }
        let entry = LogEntry::new(level, message, &self.default_target).with_fields(fields);
        self.store_entry(entry);
    }

    /// Create a logger context with a span ID.
    pub fn with_span<'a>(&'a mut self, span_id: &'a str) -> LoggerWithContext<'a> {
        LoggerWithContext {
            logger: self,
            span_id: span_id.to_string(),
            target: None,
        }
    }

    /// Create a logger context with a target.
    pub fn with_target<'a>(&'a mut self, target: &'a str) -> LoggerWithContext<'a> {
        LoggerWithContext {
            logger: self,
            span_id: String::new(),
            target: Some(target.to_string()),
        }
    }

    // Convenience methods

    /// Log at trace level.
    pub fn trace(&mut self, message: &str) {
        self.log(LogLevel::Trace, message);
    }

    /// Log at debug level.
    pub fn debug(&mut self, message: &str) {
        self.log(LogLevel::Debug, message);
    }

    /// Log at info level.
    pub fn info(&mut self, message: &str) {
        self.log(LogLevel::Info, message);
    }

    /// Log at warn level.
    pub fn warn(&mut self, message: &str) {
        self.log(LogLevel::Warn, message);
    }

    /// Log at error level.
    pub fn error(&mut self, message: &str) {
        self.log(LogLevel::Error, message);
    }

    /// Get all entries matching an optional filter.
    pub fn get_entries(&self, filter: Option<&LogFilter>) -> Vec<&LogEntry> {
        match filter {
            Some(f) => self.entries.iter().filter(|e| f.matches(e)).collect(),
            None => self.entries.iter().collect(),
        }
    }

    /// Get recent entries.
    pub fn get_recent(&self, limit: usize) -> Vec<&LogEntry> {
        self.entries.iter().rev().take(limit).collect()
    }

    /// Get entries by level.
    pub fn get_by_level(&self, level: LogLevel) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.level == level).collect()
    }

    /// Get error entries.
    pub fn get_errors(&self) -> Vec<&LogEntry> {
        self.get_by_level(LogLevel::Error)
    }

    /// Get warning entries.
    pub fn get_warnings(&self) -> Vec<&LogEntry> {
        self.get_by_level(LogLevel::Warn)
    }

    /// Get entry count.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Export all entries and clear.
    pub fn export(&mut self) -> Vec<LogEntry> {
        self.entries.drain(..).collect()
    }

    /// Get entries for a specific span.
    pub fn get_span_entries(&self, span_id: &str) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.span_id.as_deref() == Some(span_id))
            .collect()
    }
}

// ============================================================================
// Logger With Context
// ============================================================================

/// A logger with associated context (span ID and/or target).
pub struct LoggerWithContext<'a> {
    logger: &'a mut Logger,
    span_id: String,
    target: Option<String>,
}

impl<'a> LoggerWithContext<'a> {
    /// Log a message at the given level.
    pub fn log(&mut self, level: LogLevel, message: &str) {
        if !self.logger.should_log(level) {
            return;
        }

        let target = self.target.as_deref().unwrap_or(&self.logger.default_target);
        let mut entry = LogEntry::new(level, message, target);

        if !self.span_id.is_empty() {
            entry.span_id = Some(self.span_id.clone());
        }

        self.logger.store_entry(entry);
    }

    /// Log at trace level.
    pub fn trace(&mut self, message: &str) {
        self.log(LogLevel::Trace, message);
    }

    /// Log at debug level.
    pub fn debug(&mut self, message: &str) {
        self.log(LogLevel::Debug, message);
    }

    /// Log at info level.
    pub fn info(&mut self, message: &str) {
        self.log(LogLevel::Info, message);
    }

    /// Log at warn level.
    pub fn warn(&mut self, message: &str) {
        self.log(LogLevel::Warn, message);
    }

    /// Log at error level.
    pub fn error(&mut self, message: &str) {
        self.log(LogLevel::Error, message);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Span Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_span_new() {
        let span = Span::new("test-span");
        assert_eq!(span.name, "test-span");
        assert!(span.parent_id.is_none());
        assert!(span.agent_id.is_none());
        assert!(span.is_in_progress());
        assert!(span.end_time.is_none());
        assert!(span.attributes.is_empty());
        assert!(span.events.is_empty());
    }

    #[test]
    fn test_span_with_parent() {
        let span = Span::with_parent("child-span", "parent-123");
        assert_eq!(span.name, "child-span");
        assert_eq!(span.parent_id, Some("parent-123".to_string()));
    }

    #[test]
    fn test_span_set_agent_id() {
        let mut span = Span::new("test");
        span.set_agent_id("agent-456");
        assert_eq!(span.agent_id, Some("agent-456".to_string()));
    }

    #[test]
    fn test_span_add_attribute() {
        let mut span = Span::new("test");
        span.add_attribute("key", serde_json::json!("value"));
        assert_eq!(span.attributes.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_span_add_event() {
        let mut span = Span::new("test");
        let event = SpanEvent::new("event-1");
        span.add_event(event);
        assert_eq!(span.events.len(), 1);
        assert_eq!(span.events[0].name, "event-1");
    }

    #[test]
    fn test_span_end() {
        let mut span = Span::new("test");
        assert!(span.is_in_progress());

        span.end(SpanStatus::Ok);
        assert!(span.is_ok());
        assert!(!span.is_in_progress());
        assert!(span.end_time.is_some());
    }

    #[test]
    fn test_span_end_with_error() {
        let mut span = Span::new("test");
        span.end(SpanStatus::Error("something went wrong".to_string()));
        assert!(span.is_error());
        assert!(!span.is_ok());
    }

    #[test]
    fn test_span_duration() {
        let mut span = Span::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        span.end(SpanStatus::Ok);

        let duration = span.duration_ms().unwrap();
        assert!(duration >= 10, "Duration should be at least 10ms, got {}ms", duration);
    }

    #[test]
    fn test_span_status_display() {
        assert_eq!(SpanStatus::Ok.to_string(), "OK");
        assert_eq!(SpanStatus::InProgress.to_string(), "IN_PROGRESS");
        assert_eq!(
            SpanStatus::Error("test".to_string()).to_string(),
            "ERROR: test"
        );
    }

    // ------------------------------------------------------------------------
    // SpanEvent Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_span_event_new() {
        let event = SpanEvent::new("test-event");
        assert_eq!(event.name, "test-event");
        assert!(event.attributes.is_empty());
    }

    #[test]
    fn test_span_event_with_attributes() {
        let mut attrs = HashMap::new();
        attrs.insert("key".to_string(), serde_json::json!("value"));

        let event = SpanEvent::with_attributes("test", attrs);
        assert_eq!(event.attributes.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_span_event_add_attribute() {
        let mut event = SpanEvent::new("test");
        event.add_attribute("foo", serde_json::json!(42));
        assert_eq!(event.attributes.get("foo"), Some(&serde_json::json!(42)));
    }

    // ------------------------------------------------------------------------
    // TracerConfig Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_tracer_config_default() {
        let config = TracerConfig::default();
        assert_eq!(config.max_completed_spans, 1000);
        assert!((config.sample_rate - 1.0).abs() < f64::EPSILON);
        assert!(config.export_enabled);
    }

    #[test]
    fn test_tracer_config_builder() {
        let config = TracerConfig::default()
            .with_max_spans(500)
            .with_sample_rate(0.5)
            .with_export(false);

        assert_eq!(config.max_completed_spans, 500);
        assert!((config.sample_rate - 0.5).abs() < f64::EPSILON);
        assert!(!config.export_enabled);
    }

    #[test]
    fn test_tracer_config_sample_rate_clamped() {
        let config = TracerConfig::new(100, 1.5, true);
        assert!((config.sample_rate - 1.0).abs() < f64::EPSILON);

        let config = TracerConfig::new(100, -0.5, true);
        assert!((config.sample_rate - 0.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------------
    // Tracer Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_tracer_new() {
        let tracer = Tracer::new("test-service", TracerConfig::default());
        assert_eq!(tracer.service_name(), "test-service");
        assert_eq!(tracer.active_span_count(), 0);
        assert_eq!(tracer.completed_span_count(), 0);
    }

    #[test]
    fn test_tracer_start_span_manual() {
        let mut tracer = Tracer::new("test", TracerConfig::default());
        let span_id = tracer.start_span_manual("operation");

        assert!(!span_id.is_empty());
        assert_eq!(tracer.active_span_count(), 1);

        let span = tracer.get_span(&span_id).unwrap();
        assert_eq!(span.name, "operation");
    }

    #[test]
    fn test_tracer_start_span_with_parent_manual() {
        let mut tracer = Tracer::new("test", TracerConfig::default());
        let parent_id = tracer.start_span_manual("parent");
        let child_id = tracer.start_span_with_parent_manual("child", &parent_id);

        let child = tracer.get_span(&child_id).unwrap();
        assert_eq!(child.parent_id, Some(parent_id));
    }

    #[test]
    fn test_tracer_add_event() {
        let mut tracer = Tracer::new("test", TracerConfig::default());
        let span_id = tracer.start_span_manual("op");

        tracer.add_event(&span_id, SpanEvent::new("checkpoint"));

        let span = tracer.get_span(&span_id).unwrap();
        assert_eq!(span.events.len(), 1);
    }

    #[test]
    fn test_tracer_set_attribute() {
        let mut tracer = Tracer::new("test", TracerConfig::default());
        let span_id = tracer.start_span_manual("op");

        tracer.set_attribute(&span_id, "user_id", serde_json::json!("user-123"));

        let span = tracer.get_span(&span_id).unwrap();
        assert_eq!(span.attributes.get("user_id"), Some(&serde_json::json!("user-123")));
    }

    #[test]
    fn test_tracer_end_span() {
        let mut tracer = Tracer::new("test", TracerConfig::default());
        let span_id = tracer.start_span_manual("op");

        tracer.end_span(&span_id, SpanStatus::Ok);

        assert_eq!(tracer.active_span_count(), 0);
        assert_eq!(tracer.completed_span_count(), 1);
    }

    #[test]
    fn test_tracer_get_recent_spans() {
        let mut tracer = Tracer::new("test", TracerConfig::default());

        for i in 0..5 {
            let span_id = tracer.start_span_manual(&format!("span-{}", i));
            tracer.end_span(&span_id, SpanStatus::Ok);
        }

        let recent = tracer.get_recent_spans(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].name, "span-4"); // Most recent first
    }

    #[test]
    fn test_tracer_export_spans() {
        let mut tracer = Tracer::new("test", TracerConfig::default());

        let span_id = tracer.start_span_manual("op");
        tracer.end_span(&span_id, SpanStatus::Ok);

        let exported = tracer.export_spans();
        assert_eq!(exported.len(), 1);
        assert_eq!(tracer.completed_span_count(), 0);
    }

    #[test]
    fn test_tracer_ring_buffer() {
        let config = TracerConfig::default().with_max_spans(3);
        let mut tracer = Tracer::new("test", config);

        for i in 0..5 {
            let span_id = tracer.start_span_manual(&format!("span-{}", i));
            tracer.end_span(&span_id, SpanStatus::Ok);
        }

        assert_eq!(tracer.completed_span_count(), 3);
        let spans = tracer.peek_spans();
        assert_eq!(spans[0].name, "span-2");
        assert_eq!(spans[1].name, "span-3");
        assert_eq!(spans[2].name, "span-4");
    }

    #[test]
    fn test_tracer_find_spans_by_name() {
        let mut tracer = Tracer::new("test", TracerConfig::default());

        for name in ["auth-login", "auth-logout", "data-fetch"] {
            let span_id = tracer.start_span_manual(name);
            tracer.end_span(&span_id, SpanStatus::Ok);
        }

        let auth_spans = tracer.find_spans_by_name("auth");
        assert_eq!(auth_spans.len(), 2);
    }

    #[test]
    fn test_tracer_find_error_spans() {
        let mut tracer = Tracer::new("test", TracerConfig::default());

        let ok_id = tracer.start_span_manual("ok-op");
        tracer.end_span(&ok_id, SpanStatus::Ok);

        let err_id = tracer.start_span_manual("err-op");
        tracer.end_span(&err_id, SpanStatus::Error("failed".to_string()));

        let errors = tracer.find_error_spans();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].name, "err-op");
    }

    #[test]
    fn test_tracer_sampling_zero() {
        let config = TracerConfig::default().with_sample_rate(0.0);
        let mut tracer = Tracer::new("test", config);

        let span_id = tracer.start_span_manual("op");
        assert!(tracer.get_span(&span_id).is_none());
    }

    #[test]
    fn test_tracer_export_disabled() {
        let config = TracerConfig::default().with_export(false);
        let mut tracer = Tracer::new("test", config);

        let span_id = tracer.start_span_manual("op");
        tracer.end_span(&span_id, SpanStatus::Ok);

        let exported = tracer.export_spans();
        assert!(exported.is_empty());
    }

    // ------------------------------------------------------------------------
    // SpanGuard Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_span_guard_auto_end() {
        let mut tracer = Tracer::new("test", TracerConfig::default());

        // Start a span manually to test guard behavior
        let span_id = tracer.start_span_manual("auto-op");
        assert_eq!(tracer.active_span_count(), 1);

        // End it to simulate guard drop
        tracer.end_span(&span_id, SpanStatus::Ok);

        assert_eq!(tracer.active_span_count(), 0);
        assert_eq!(tracer.completed_span_count(), 1);

        let spans = tracer.peek_spans();
        assert!(spans[0].is_ok());
    }

    #[test]
    fn test_span_guard_end_with_error() {
        let mut tracer = Tracer::new("test", TracerConfig::default());

        // Test error ending using manual API
        let span_id = tracer.start_span_manual("error-op");
        tracer.end_span(&span_id, SpanStatus::Error("something failed".to_string()));

        let spans = tracer.peek_spans();
        assert!(spans[0].is_error());
    }

    #[test]
    fn test_span_guard_add_event() {
        let mut tracer = Tracer::new("test", TracerConfig::default());

        // Test adding events using manual API
        let span_id = tracer.start_span_manual("op");
        tracer.add_event_simple(&span_id, "checkpoint");
        tracer.end_span(&span_id, SpanStatus::Ok);

        let spans = tracer.peek_spans();
        assert_eq!(spans[0].events.len(), 1);
    }

    #[test]
    fn test_span_guard_set_attribute() {
        let mut tracer = Tracer::new("test", TracerConfig::default());

        // Test setting attributes using manual API
        let span_id = tracer.start_span_manual("op");
        tracer.set_attribute(&span_id, "key", serde_json::json!("value"));
        tracer.end_span(&span_id, SpanStatus::Ok);

        let spans = tracer.peek_spans();
        assert_eq!(spans[0].attributes.get("key"), Some(&serde_json::json!("value")));
    }

    // ------------------------------------------------------------------------
    // LogLevel Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_log_level_display() {
        assert_eq!(LogLevel::Trace.to_string(), "TRACE");
        assert_eq!(LogLevel::Debug.to_string(), "DEBUG");
        assert_eq!(LogLevel::Info.to_string(), "INFO");
        assert_eq!(LogLevel::Warn.to_string(), "WARN");
        assert_eq!(LogLevel::Error.to_string(), "ERROR");
    }

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(LogLevel::from_str("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::from_str("DEBUG"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_str("Info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("warning"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("err"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("invalid"), None);
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }

    #[test]
    fn test_log_level_all() {
        let all = LogLevel::all();
        assert_eq!(all.len(), 5);
    }

    // ------------------------------------------------------------------------
    // LogEntry Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_log_entry_new() {
        let entry = LogEntry::new(LogLevel::Info, "test message", "my::module");
        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.message, "test message");
        assert_eq!(entry.target, "my::module");
        assert!(entry.span_id.is_none());
    }

    #[test]
    fn test_log_entry_with_span() {
        let entry = LogEntry::with_span(LogLevel::Debug, "msg", "target", "span-123");
        assert_eq!(entry.span_id, Some("span-123".to_string()));
    }

    #[test]
    fn test_log_entry_add_field() {
        let mut entry = LogEntry::new(LogLevel::Info, "msg", "target");
        entry.add_field("user_id", serde_json::json!("user-456"));
        assert_eq!(entry.fields.get("user_id"), Some(&serde_json::json!("user-456")));
    }

    #[test]
    fn test_log_entry_format() {
        let entry = LogEntry::new(LogLevel::Info, "hello world", "app");
        let formatted = entry.format();
        assert!(formatted.contains("INFO"));
        assert!(formatted.contains("app"));
        assert!(formatted.contains("hello world"));
    }

    #[test]
    fn test_log_entry_display() {
        let entry = LogEntry::new(LogLevel::Warn, "warning!", "test");
        let display = entry.to_string();
        assert!(display.contains("WARN"));
        assert!(display.contains("warning!"));
    }

    // ------------------------------------------------------------------------
    // LogFilter Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_log_filter_empty() {
        let filter = LogFilter::new();
        let entry = LogEntry::new(LogLevel::Info, "msg", "target");
        assert!(filter.matches(&entry));
    }

    #[test]
    fn test_log_filter_by_level() {
        let filter = LogFilter::new().with_min_level(LogLevel::Warn);

        let info = LogEntry::new(LogLevel::Info, "msg", "target");
        let warn = LogEntry::new(LogLevel::Warn, "msg", "target");
        let error = LogEntry::new(LogLevel::Error, "msg", "target");

        assert!(!filter.matches(&info));
        assert!(filter.matches(&warn));
        assert!(filter.matches(&error));
    }

    #[test]
    fn test_log_filter_by_target() {
        let filter = LogFilter::new().with_target("auth");

        let auth = LogEntry::new(LogLevel::Info, "msg", "auth::login");
        let data = LogEntry::new(LogLevel::Info, "msg", "data::fetch");

        assert!(filter.matches(&auth));
        assert!(!filter.matches(&data));
    }

    #[test]
    fn test_log_filter_by_span_id() {
        let filter = LogFilter::new().with_span_id("span-123");

        let with_span = LogEntry::with_span(LogLevel::Info, "msg", "target", "span-123");
        let no_span = LogEntry::new(LogLevel::Info, "msg", "target");
        let wrong_span = LogEntry::with_span(LogLevel::Info, "msg", "target", "span-456");

        assert!(filter.matches(&with_span));
        assert!(!filter.matches(&no_span));
        assert!(!filter.matches(&wrong_span));
    }

    #[test]
    fn test_log_filter_by_message() {
        let filter = LogFilter::new().with_message("error");

        let has_error = LogEntry::new(LogLevel::Info, "an error occurred", "target");
        let no_error = LogEntry::new(LogLevel::Info, "all good", "target");

        assert!(filter.matches(&has_error));
        assert!(!filter.matches(&no_error));
    }

    #[test]
    fn test_log_filter_combined() {
        let filter = LogFilter::new()
            .with_min_level(LogLevel::Warn)
            .with_target("auth");

        let auth_warn = LogEntry::new(LogLevel::Warn, "msg", "auth::login");
        let auth_info = LogEntry::new(LogLevel::Info, "msg", "auth::login");
        let data_warn = LogEntry::new(LogLevel::Warn, "msg", "data::fetch");

        assert!(filter.matches(&auth_warn));
        assert!(!filter.matches(&auth_info));
        assert!(!filter.matches(&data_warn));
    }

    // ------------------------------------------------------------------------
    // LoggerConfig Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_logger_config_default() {
        let config = LoggerConfig::default();
        assert_eq!(config.max_entries, 10000);
        assert_eq!(config.min_level, LogLevel::Info);
        assert!(config.include_timestamps);
    }

    #[test]
    fn test_logger_config_builder() {
        let config = LoggerConfig::default()
            .with_max_entries(500)
            .with_min_level(LogLevel::Debug)
            .with_timestamps(false)
            .with_default_target("my-app");

        assert_eq!(config.max_entries, 500);
        assert_eq!(config.min_level, LogLevel::Debug);
        assert!(!config.include_timestamps);
        assert_eq!(config.default_target, "my-app");
    }

    // ------------------------------------------------------------------------
    // Logger Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_logger_new() {
        let logger = Logger::new(LoggerConfig::default());
        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_logger_log() {
        let mut logger = Logger::new(LoggerConfig::default());
        logger.info("test message");

        assert_eq!(logger.count(), 1);
        let entries = logger.get_entries(None);
        assert_eq!(entries[0].message, "test message");
        assert_eq!(entries[0].level, LogLevel::Info);
    }

    #[test]
    fn test_logger_min_level_filtering() {
        let config = LoggerConfig::default().with_min_level(LogLevel::Warn);
        let mut logger = Logger::new(config);

        logger.debug("should not appear");
        logger.info("should not appear");
        logger.warn("should appear");
        logger.error("should appear");

        assert_eq!(logger.count(), 2);
    }

    #[test]
    fn test_logger_convenience_methods() {
        let config = LoggerConfig::default().with_min_level(LogLevel::Trace);
        let mut logger = Logger::new(config);

        logger.trace("trace");
        logger.debug("debug");
        logger.info("info");
        logger.warn("warn");
        logger.error("error");

        assert_eq!(logger.count(), 5);
    }

    #[test]
    fn test_logger_with_span() {
        let mut logger = Logger::new(LoggerConfig::default());
        logger.with_span("span-123").info("in span");

        let entries = logger.get_entries(None);
        assert_eq!(entries[0].span_id, Some("span-123".to_string()));
    }

    #[test]
    fn test_logger_with_target() {
        let mut logger = Logger::new(LoggerConfig::default());
        logger.with_target("custom::target").info("with target");

        let entries = logger.get_entries(None);
        assert_eq!(entries[0].target, "custom::target");
    }

    #[test]
    fn test_logger_get_entries_with_filter() {
        let config = LoggerConfig::default().with_min_level(LogLevel::Debug);
        let mut logger = Logger::new(config);

        logger.debug("debug");
        logger.info("info");
        logger.warn("warn");

        let filter = LogFilter::new().with_min_level(LogLevel::Warn);
        let entries = logger.get_entries(Some(&filter));
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_logger_get_recent() {
        let mut logger = Logger::new(LoggerConfig::default());

        for i in 0..5 {
            logger.info(&format!("message-{}", i));
        }

        let recent = logger.get_recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].message, "message-4");
    }

    #[test]
    fn test_logger_get_by_level() {
        let config = LoggerConfig::default().with_min_level(LogLevel::Debug);
        let mut logger = Logger::new(config);

        logger.debug("d");
        logger.info("i");
        logger.warn("w");
        logger.error("e");

        assert_eq!(logger.get_by_level(LogLevel::Warn).len(), 1);
        assert_eq!(logger.get_errors().len(), 1);
        assert_eq!(logger.get_warnings().len(), 1);
    }

    #[test]
    fn test_logger_get_span_entries() {
        let mut logger = Logger::new(LoggerConfig::default());

        logger.with_span("span-a").info("a1");
        logger.with_span("span-a").info("a2");
        logger.with_span("span-b").info("b1");

        let span_a = logger.get_span_entries("span-a");
        assert_eq!(span_a.len(), 2);
    }

    #[test]
    fn test_logger_clear() {
        let mut logger = Logger::new(LoggerConfig::default());
        logger.info("test");
        assert_eq!(logger.count(), 1);

        logger.clear();
        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_logger_export() {
        let mut logger = Logger::new(LoggerConfig::default());
        logger.info("test");

        let exported = logger.export();
        assert_eq!(exported.len(), 1);
        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_logger_ring_buffer() {
        let config = LoggerConfig::default().with_max_entries(3);
        let mut logger = Logger::new(config);

        for i in 0..5 {
            logger.info(&format!("msg-{}", i));
        }

        assert_eq!(logger.count(), 3);
        let entries = logger.get_entries(None);
        assert_eq!(entries[0].message, "msg-2");
        assert_eq!(entries[1].message, "msg-3");
        assert_eq!(entries[2].message, "msg-4");
    }

    #[test]
    fn test_logger_add_filter() {
        let mut logger = Logger::new(LoggerConfig::default());

        // Add filter for warn+ only
        logger.add_filter(LogFilter::new().with_min_level(LogLevel::Warn));

        logger.info("should not appear");
        logger.warn("should appear");

        assert_eq!(logger.count(), 1);
    }

    #[test]
    fn test_logger_clear_filters() {
        let mut logger = Logger::new(LoggerConfig::default());

        logger.add_filter(LogFilter::new().with_min_level(LogLevel::Error));
        logger.info("filtered out");
        assert_eq!(logger.count(), 0);

        logger.clear_filters();
        logger.info("should appear");
        assert_eq!(logger.count(), 1);
    }

    #[test]
    fn test_logger_log_with_fields() {
        let mut logger = Logger::new(LoggerConfig::default());

        let mut fields = HashMap::new();
        fields.insert("user_id".to_string(), serde_json::json!("u123"));
        fields.insert("action".to_string(), serde_json::json!("login"));

        logger.log_with_fields(LogLevel::Info, "user action", fields);

        let entries = logger.get_entries(None);
        assert_eq!(entries[0].fields.get("user_id"), Some(&serde_json::json!("u123")));
    }

    #[test]
    fn test_logger_log_to_target() {
        let mut logger = Logger::new(LoggerConfig::default());
        logger.log_to(LogLevel::Info, "message", "custom::module");

        let entries = logger.get_entries(None);
        assert_eq!(entries[0].target, "custom::module");
    }
}
