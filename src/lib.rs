//! # Project Panpsychism
//!
//! 🪄 The Sorcerer's Wand — Transform your words into creation.
//!
//! ## Overview
//!
//! Project Panpsychism implements a sophisticated prompt library orchestration system
//! using the Sorcerer's Wand metaphor:
//!
//! - **Sorcerer** (you) speaks incantations (queries/intents)
//! - **Grimoire** contains your spells (prompt library)
//! - **Wand** (this tool) channels the magic
//! - **Creation** is the result of successful spellwork
//!
//! ## Philosophical Foundation
//!
//! Grounded in Spinoza's philosophical framework:
//!
//! - **Conatus**: The system's drive toward self-preservation and optimal response
//! - **Natura**: Natural alignment between human thought and AI understanding
//! - **Ratio**: Logical validation ensuring coherent prompt synthesis
//! - **Laetitia**: Joy optimization through successful thought transfer
//!
//! ## Architecture
//!
//! ```text
//! 🧙 Sorcerer (You)
//!        |
//!        | "Incantation" (Your intent/query)
//!        v
//! +------------------+
//! |   🪄 The Wand    |  <-- Orchestrator: channels your magic
//! +------------------+
//!        |
//!   +----+----+
//!   |         |
//!   v         v
//! +------+ +--------+
//! |Search| |Indexer |  <-- 📜 Grimoire: find the right spells
//! +------+ +--------+
//!        |
//!        v
//! +------------------+
//! |   Synthesizer    |  <-- ⚗️ Combine spells into powerful magic
//! +------------------+
//!        |
//!        v
//! +------------------+
//! |    Validator     |  <-- 🏛️ Spinoza's blessing (philosophical validation)
//! +------------------+
//!        |
//!        v
//! +------------------+
//! |    Corrector     |  <-- ✨ Refine the spell if needed
//! +------------------+
//!        |
//!        v
//!   🎇 Creation (Final Response)
//! ```
//!
//! ## Privacy Tiers
//!
//! All data is classified into privacy tiers:
//!
//! - **Public**: Open prompts, shareable templates
//! - **Internal**: Organization-specific prompts
//! - **Confidential**: Personal thought patterns
//! - **Restricted**: Sensitive cognitive data
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use panpsychism::{Config, PrivacyTier};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let config = Config::from_env()?;
//!
//!     // Summon the wand (orchestrator)
//!     let orchestrator = panpsychism::orchestrator::Orchestrator::new(config).await?;
//!
//!     // Cast your spell (process user intent)
//!     let response = orchestrator.process("How do I implement authentication?").await?;
//!
//!     println!("✨ Creation: {}", response);
//!     Ok(())
//! }
//! ```

// Module declarations
pub mod adapter;
pub mod api;
pub mod auditor;
pub mod bus;
pub mod cache;
pub mod config;
pub mod consciousness;
pub mod constants;
pub mod contextualizer;
pub mod corrector;
pub mod debugger;
pub mod documenter;
pub mod enhancer;
pub mod enricher;
pub mod error;
pub mod evaluator;
pub mod expander;
pub mod federator;
pub mod formatter;
pub mod gemini;
pub mod harmonizer;
pub mod indexer;
pub mod learner;
pub mod llm;
pub mod localizer;
pub mod monitor;
pub mod orchestrator;
pub mod predictor;
pub mod privacy;
pub mod recovery;
pub mod recoverer;
pub mod recommender;
pub mod sanitizer;
pub mod search;
pub mod summarizer;
pub mod synthesizer;
pub mod tracing_setup;
pub mod validator;
pub mod personalizer;
pub mod rate_limiter;
pub mod refactorer;
pub mod templater;
pub mod transcender;
pub mod evolver;
pub mod memory;
pub mod settings;
pub mod telemetry;
pub mod cli;

// ============================================================================
// Type Aliases
// ============================================================================
// Common type aliases to reduce duplication and improve readability.

use std::collections::HashSet;

/// Type alias for prompt identifiers (typically UUID or file path).
pub type PromptId = String;

/// Type alias for scores and thresholds (0.0 - 1.0 range).
pub type Score = f64;

/// Type alias for tag collections used in prompt metadata.
pub type TagSet = HashSet<String>;

// Re-exports for convenient access
pub use adapter::{
    AdaptedPrompt, AdapterAgent, AdapterAgentBuilder, AdapterConfig, TargetFormat,
    TransformationRule, ValidationResult,
};
pub use api::{
    // WebSocket types
    WebSocketServer, WebSocketServerBuilder, WebSocketConfig, Connection, ConnectionId,
    ConnectionState, ClientInfo, ClientMessage, ServerMessage, QueryResult, QueryRequest,
    QueryHandler, MessageHandler, WebSocketStats,
    // REST API types
    ApiServer, ApiServerBuilder, ApiConfig, Router, Route,
    Method, Status, Request, Response, ApiError, RouteHandler,
    Middleware, CorsMiddleware, RateLimitMiddleware, LoggingMiddleware, AuthMiddleware,
    ApiState, RestAgentInfo, RestAgentStatus, ServerMetrics,
    RestRateLimitConfig, AuthConfig,
};
pub use auditor::{
    AuditFinding, AuditLevel, AuditReport, AuditSeverity, AuditorAgent, AuditorAgentBuilder,
    AuditorConfig, PolicyRule, PolicySet,
};
pub use bus::{
    AgentBus, AgentId, AgentInfo, AgentStatus as BusAgentStatus, BusConfig, BusParticipant,
    BusStats, EventType, HeartbeatStatus, Message, MessageId, Topic,
};
pub use cache::{CacheConfig, CacheManager, CacheStats, CachedSearchEngine};
pub use config::{
    AnalyzeConfig, Config, ContentTypeMapping, InputSource,
    OutputConfig as AnalyzeOutputConfig, OutputDestination as AnalyzeOutputDestination,
};
pub use constants::*;
pub use contextualizer::{
    ContextConfig, ContextError, ContextWindow, ContextualizedQuery, ContextualizerAgent,
    Interaction, Memory, MemoryStats, SessionMemory,
};
pub use error::{Error, Result};
pub use expander::{
    Alternative, ContextInfo, Definition, Example, ExampleComplexity, ExpandedContent,
    ExpanderAgent, ExpanderAgentBuilder, ExpanderConfig, ExpansionType,
};
pub use federator::{
    Capability, FederatedResponse, FederationConfig, FederationStats, FederationStatsSnapshot,
    FederatorAgent, FederatorAgentBuilder, Instance, InstanceHealth, LoadBalanceStrategy,
};
pub use formatter::{
    FormatTemplate, FormattedOutput, FormatterAgent, FormatterAgentBuilder, OutputFormat,
};
pub use privacy::{PrivacyConfig, PrivacyTier};
pub use recovery::{
    is_rate_limit_error, is_retryable_error, with_retry, with_timeout, with_timeout_optional,
    with_timeouts, CircuitBreaker, CircuitState, FallbackConfig, FallbackResult, PartialResults,
    RetryConfig, ShutdownManager, ShutdownState,
};
pub use summarizer::{CompressionLevel, KeyPoint, Summary, SummarizerAgent, SummarizerConfig};
pub use predictor::{
    Prediction, PredictionResult, PredictionStrategy, PredictorAgent, PredictorAgentBuilder,
    PredictorConfig,
};
pub use recommender::{
    Recommendation, RecommendationConfig, RecommendationSet, RecommendationStrategy,
    RecommenderAgent, RecommenderAgentBuilder,
};
pub use tracing_setup::{setup_logging, should_use_json};
pub use debugger::{
    Complexity, DebugReport, DebuggerAgent, DebuggerAgentBuilder, Fix, Issue, IssueCategory,
    RootCause, Severity,
};
pub use evaluator::{
    EvaluationDimensions, EvaluationLevel, EvaluationReport, EvaluatorAgent, EvaluatorAgentBuilder,
    EvaluatorConfig, Improvement, Priority,
};
pub use learner::{
    KnowledgeUpdate, LearnerAgent, LearnerAgentBuilder, LearnerConfig, LearningInsights, Pattern,
    PatternType, PerformanceTrends,
};
pub use enhancer::{
    EnhancedPrompt, EnhancerAgent, EnhancerAgentBuilder, EnhancerConfig, EnhancementGoal,
    EnhancementStep, Improvement as EnhancerImprovement, QualityDimension,
};
pub use personalizer::{
    Adaptation, AdaptationType, DomainPreference, ExpertiseLevel, LanguageStyle,
    PersonalizedPrompt, PersonalizerAgent, PersonalizerAgentBuilder, PersonalizerConfig,
    ProfileSummary, Tone, UserProfile, VerbosityLevel, VocabularyPreferences,
};
pub use localizer::{
    CulturalCategory, CulturalNote, CulturalSeverity, Locale, LocaleAlternative,
    LocalizedPrompt, LocalizerAgent, LocalizerAgentBuilder, LocalizerConfig,
};
pub use consciousness::{
    AgentState, AgentStatus, ConsciousnessAgent, ConsciousnessAgentBuilder, ConsciousnessConfig,
    ConsciousnessReport, HealthLevel, Insight, InsightCategory, OperatingMode, Process,
    SelfAssessment, SystemState,
};
pub use monitor::{
    AgentHealth, Alert, AlertLevel, HealthReport, HealthStatus, MonitorAgent,
    MonitorAgentBuilder, MonitorConfig, SystemMetrics,
};
pub use enricher::{
    Citation, CrossRef, EnrichedPrompt, EnricherAgent, EnricherAgentBuilder, EnricherConfig,
    EnrichmentExample, EnrichmentMetadata, EnrichmentStrategy,
};
pub use rate_limiter::{
    ClientId, RateLimitConfig, RateLimitDecision, RateLimitStrategy, RateLimiterAgent,
    RateLimiterAgentBuilder, UsageStats,
};
pub use recoverer::{
    DegradeLevel, ErrorContext, RecovererAgent, RecovererAgentBuilder, RecoveryAction,
    RecoveryConfig, RecoveryStats, RecoveryStatsSnapshot, RecoveryStrategy,
};
pub use sanitizer::{
    Normalization, NormalizationType, Removal, RemovalCategory, SanitizationRules,
    SanitizedInput, SanitizerAgent, SanitizerAgentBuilder, SanitizerConfig,
};
pub use documenter::{
    Audience, DocConfig, DocFormat, DocSection, Documentation, DocumenterAgent,
    DocumenterAgentBuilder,
};
pub use refactorer::{
    Constraint, PromptMetrics, RefactoredPrompt, RefactorerAgent, RefactorerAgentBuilder,
    RefactorerConfig, RefactoringChange, RefactoringGoal, RefactoringType,
};
pub use templater::{
    InstantiatedPrompt, PromptTemplate, TemplateRegistry, TemplateVariable, TemplaterAgent,
    TemplaterAgentBuilder, TemplaterConfig, VariableType,
};
pub use harmonizer::{
    Adjustment, AdjustmentType, Conflict, ConflictType, HarmonizerAgent, HarmonizerAgentBuilder,
    HarmonizerConfig, HarmonyReport, ResourceAllocation,
};
pub use transcender::{
    Constraint as TranscenderConstraint, FallbackPath, OptimizationTarget, OrchestrationPlan,
    Phase, PhaseResult, TranscendentGoal, TranscendentResponse, TranscenderAgent,
    TranscenderAgentBuilder, TranscenderConfig,
};
pub use evolver::{
    Candidate, EvolutionGoal, EvolutionMetric, EvolutionReport, EvolverAgent,
    EvolverAgentBuilder, EvolverConfig, Generation, Mutation, MutationType,
};
pub use memory::{
    Memorable, Memory as PersistentMemory, MemoryConfig, MemoryLayer, MemoryLayerBuilder,
    MemoryScope, MemoryStats as PersistentMemoryStats,
};
pub use settings::{
    AgentSettings, BusSettings, Environment, GeneralSettings, LogFormat, LogLevel,
    LoggingSettings, MemorySettings as SettingsMemorySettings, Settings, SettingsBuilder,
    SettingsError, SettingsResult,
};
pub use telemetry::{
    LogEntry, LogFilter, LogLevel as TelemetryLogLevel, Logger, LoggerConfig,
    Span, SpanEvent, SpanStatus, Tracer, TracerConfig,
};
pub use llm::{
    // Core traits
    LLMClient, OllamaLLMClient,
    // Shared types
    MessageRole, ChatMessage, CompletionResponse, ChatResponse,
    GenerationOptions, ModelInfo, HealthCheck,
    // OpenAI types
    OpenAIClient, OpenAIClientBuilder, OpenAIConfig, OpenAIModel,
    OpenAIMessage, OpenAIRequest, OpenAIResponse,
    OpenAIChoice, OpenAIUsage, OpenAIStreamChunk, OpenAIStreamChoice, OpenAIStreamDelta,
    OpenAIErrorCategory, categorize_error,
    OpenAIRateLimiter, OpenAIUsageStats, OpenAIUsageStatsSnapshot,
    estimate_tokens, estimate_messages_tokens,
    DEFAULT_OPENAI_ENDPOINT, DEFAULT_TIMEOUT_SECS, DEFAULT_MAX_RETRIES,
    // Ollama types
    OllamaClient, OllamaClientBuilder, OllamaConfig, OllamaModel,
    // Anthropic types
    AnthropicClient, AnthropicClientBuilder, AnthropicConfig, AnthropicModel,
    AnthropicMessage, AnthropicRequest, AnthropicResponse,
    AnthropicUsage, ContentBlock, StopReason,
    AnthropicErrorKind, AnthropicModelInfo,
    DEFAULT_ANTHROPIC_ENDPOINT, DEFAULT_ANTHROPIC_VERSION,
    // Router types
    LLMProvider, RoutingStrategy, ProviderConfig, LLMRouter, LLMRouterBuilder,
    CircuitBreaker as RouterCircuitBreaker, CircuitState as RouterCircuitState,
    CostTracker, CostSnapshot, ProviderHealthStatus, RouterStats,
};
pub use cli::{
    AgentInfo as CliAgentInfo, AgentStatus as CliAgentStatus, CliApp, CliAppBuilder, CliConfig,
    CliConfigBuilder, Command, CommandResult, ConfigAction, HistoryEntry,
    OutputFormat as CliOutputFormat,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");
