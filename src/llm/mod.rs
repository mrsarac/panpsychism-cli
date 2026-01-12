//! # LLM Router Module for Project Panpsychism
//!
//! Provides a unified interface for multiple LLM providers with automatic
//! provider selection, fallback, and load balancing.
//!
//! ## Supported Providers
//!
//! - **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-4, GPT-3.5-turbo
//! - **Ollama**: Local model runner (llama3.2, mistral, phi3, etc.)
//! - **Gemini**: (via gemini.rs in parent module)
//!
//! ## Features
//!
//! - Unified LLMClient trait for provider abstraction
//! - Automatic retry with exponential backoff
//! - Rate limiting (TPM/RPM) per provider
//! - Usage statistics and cost tracking
//! - Streaming support via SSE
//! - Local-first option with Ollama
//!
//! ## Quick Start - OpenAI
//!
//! ```rust,ignore
//! use panpsychism::llm::openai::{OpenAIClient, OpenAIMessage};
//!
//! let client = OpenAIClient::builder()
//!     .api_key("sk-...")
//!     .build()?;
//!
//! let messages = vec![OpenAIMessage::user("Hello!")];
//! let response = client.chat_with_retry(messages).await?;
//! ```
//!
//! ## Quick Start - Ollama (Local)
//!
//! ```rust,ignore
//! use panpsychism::llm::{OllamaClient, OllamaLLMClient};
//!
//! let client = OllamaClient::builder()
//!     .endpoint("http://localhost:11434")
//!     .default_model("llama3.2")
//!     .build()
//!     .await?;
//!
//! let response = client.complete("Explain quantum computing").await?;
//! println!("{}", response.content);
//! ```

pub mod anthropic;
pub mod ollama;
pub mod openai;
pub mod router;

use std::collections::HashMap;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::Result;

// ============================================================================
// RE-EXPORTS - OpenAI
// ============================================================================

pub use openai::{
    categorize_error, estimate_messages_tokens, estimate_tokens, OpenAIChoice, OpenAIClient,
    OpenAIClientBuilder, OpenAIConfig, OpenAIErrorCategory, OpenAIMessage, OpenAIModel,
    OpenAIRateLimiter, OpenAIRequest, OpenAIResponse, OpenAIStreamChunk, OpenAIStreamChoice,
    OpenAIStreamDelta, OpenAIUsage, OpenAIUsageStats, OpenAIUsageStatsSnapshot,
    DEFAULT_MAX_RETRIES, DEFAULT_OPENAI_ENDPOINT, DEFAULT_TIMEOUT_SECS,
};

// ============================================================================
// RE-EXPORTS - Ollama
// ============================================================================

pub use ollama::{OllamaClient, OllamaClientBuilder, OllamaConfig, OllamaModel};

// ============================================================================
// RE-EXPORTS - Anthropic
// ============================================================================

pub use anthropic::{
    AnthropicClient, AnthropicClientBuilder, AnthropicConfig, AnthropicErrorKind,
    AnthropicMessage, AnthropicModel, AnthropicRequest, AnthropicResponse, AnthropicUsage,
    ContentBlock, ModelInfo as AnthropicModelInfo, StopReason, DEFAULT_ANTHROPIC_ENDPOINT,
    DEFAULT_ANTHROPIC_VERSION,
};

// ============================================================================
// RE-EXPORTS - Router
// ============================================================================

pub use router::{
    CircuitBreaker, CircuitState, CostSnapshot, CostTracker, LLMProvider, LLMRouter,
    LLMRouterBuilder, ProviderConfig, ProviderHealthStatus, RouterStats, RoutingStrategy,
};

// ============================================================================
// SHARED TYPES FOR OLLAMA
// ============================================================================

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message setting context and behavior.
    System,
    /// User message (human input).
    #[default]
    User,
    /// Assistant message (model response).
    Assistant,
}

impl MessageRole {
    /// Returns the string representation of the role.
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender.
    pub role: MessageRole,
    /// Content of the message.
    pub content: String,
    /// Optional name for the message sender.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            name: None,
        }
    }

    /// Create a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
        }
    }

    /// Create a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            name: None,
        }
    }

    /// Set an optional name for this message.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Options for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// Temperature for sampling (0.0 - 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Stop sequences that terminate generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Random seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Frequency penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            max_tokens: None,
            stop: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

impl GenerationOptions {
    /// Create options optimized for creative tasks.
    pub fn creative() -> Self {
        Self {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(50),
            ..Default::default()
        }
    }

    /// Create options optimized for deterministic/factual tasks.
    pub fn deterministic() -> Self {
        Self {
            temperature: Some(0.1),
            top_p: Some(0.8),
            top_k: Some(20),
            ..Default::default()
        }
    }

    /// Create options optimized for code generation.
    pub fn code() -> Self {
        Self {
            temperature: Some(0.2),
            top_p: Some(0.9),
            top_k: Some(30),
            ..Default::default()
        }
    }
}

/// Response from a completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Generated content.
    pub content: String,
    /// Model used for generation.
    pub model: String,
    /// Whether generation is complete.
    pub done: bool,
    /// Number of tokens in the prompt.
    pub prompt_tokens: Option<u32>,
    /// Number of tokens generated.
    pub completion_tokens: Option<u32>,
    /// Total tokens used.
    pub total_tokens: Option<u32>,
    /// Time taken for generation in milliseconds.
    pub duration_ms: Option<u64>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CompletionResponse {
    /// Create a new completion response.
    pub fn new(content: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            model: model.into(),
            done: true,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            duration_ms: None,
            metadata: HashMap::new(),
        }
    }
}

/// Response from a chat request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The assistant's response message.
    pub message: ChatMessage,
    /// Model used for generation.
    pub model: String,
    /// Whether generation is complete.
    pub done: bool,
    /// Number of tokens in the prompt.
    pub prompt_tokens: Option<u32>,
    /// Number of tokens generated.
    pub completion_tokens: Option<u32>,
    /// Total tokens used.
    pub total_tokens: Option<u32>,
    /// Time taken for generation in milliseconds.
    pub duration_ms: Option<u64>,
}

impl ChatResponse {
    /// Get the content of the response message.
    pub fn content(&self) -> &str {
        &self.message.content
    }
}

/// Information about an available model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name/identifier.
    pub name: String,
    /// Model size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Model family (e.g., "llama", "mistral").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Parameter count (e.g., "7B", "13B").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<String>,
    /// Quantization level (e.g., "Q4_0", "Q8_0").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Context window size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Whether this model supports chat/conversation.
    #[serde(default)]
    pub supports_chat: bool,
    /// Whether this model supports embeddings.
    #[serde(default)]
    pub supports_embeddings: bool,
}

impl ModelInfo {
    /// Create a new model info.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            size: None,
            family: None,
            parameters: None,
            quantization: None,
            context_length: None,
            supports_chat: true,
            supports_embeddings: false,
        }
    }
}

/// Health check result for an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Whether the provider is healthy.
    pub healthy: bool,
    /// Provider name.
    pub provider: String,
    /// Endpoint being checked.
    pub endpoint: String,
    /// Response time in milliseconds.
    pub latency_ms: Option<u64>,
    /// Error message if unhealthy.
    pub error: Option<String>,
    /// Number of available models.
    pub model_count: Option<usize>,
}

// ============================================================================
// OLLAMA LLM CLIENT TRAIT
// ============================================================================

/// Trait for Ollama-style LLM client implementations.
///
/// This trait provides a more feature-rich interface specifically designed
/// for local model providers like Ollama.
#[async_trait]
pub trait OllamaLLMClient: Send + Sync {
    /// Get the provider name (e.g., "ollama", "gemini", "openai").
    fn provider(&self) -> &'static str;

    /// Get the default model name.
    fn default_model(&self) -> &str;

    /// Complete a single prompt.
    async fn complete(&self, prompt: &str) -> Result<CompletionResponse>;

    /// Complete a prompt with custom options.
    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<CompletionResponse>;

    /// Complete a prompt with a specific model.
    async fn complete_with_model(
        &self,
        prompt: &str,
        model: &str,
        options: &GenerationOptions,
    ) -> Result<CompletionResponse>;

    /// Send a chat conversation and get a response.
    async fn chat(&self, messages: &[ChatMessage]) -> Result<ChatResponse>;

    /// Send a chat conversation with custom options.
    async fn chat_with_options(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
    ) -> Result<ChatResponse>;

    /// List available models.
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Check if the provider is healthy and available.
    async fn ollama_health_check(&self) -> Result<HealthCheck>;

    /// Get information about a specific model.
    async fn model_info(&self, model: &str) -> Result<ModelInfo>;
}

// ============================================================================
// ORIGINAL LLM CLIENT TRAIT (for OpenAI compatibility)
// ============================================================================

/// Trait for LLM client implementations.
///
/// This trait provides a unified interface for different LLM providers,
/// enabling provider-agnostic code and easy switching between providers.
///
/// # Example
///
/// ```rust,ignore
/// use panpsychism::llm::LLMClient;
///
/// async fn use_any_llm<C: LLMClient>(client: &C, prompt: &str) -> Result<String> {
///     client.complete(prompt).await
/// }
/// ```
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Complete a simple text prompt.
    async fn complete(&self, prompt: &str) -> Result<String>;

    /// Complete a prompt with automatic retry on transient failures.
    async fn complete_with_retry(&self, prompt: &str) -> Result<String>;

    /// Stream a completion response token by token.
    async fn complete_streaming(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>;

    /// Check if the provider's API endpoint is healthy.
    async fn health_check(&self) -> Result<bool>;

    /// Get the provider name (e.g., "openai", "gemini").
    fn provider_name(&self) -> &'static str;

    /// Get the current model identifier.
    fn model_id(&self) -> &str;
}

/// Implement LLMClient trait for OpenAIClient
#[async_trait]
impl LLMClient for OpenAIClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        OpenAIClient::complete(self, prompt).await
    }

    async fn complete_with_retry(&self, prompt: &str) -> Result<String> {
        OpenAIClient::complete_with_retry(self, prompt).await
    }

    async fn complete_streaming(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        OpenAIClient::complete_streaming(self, prompt).await
    }

    async fn health_check(&self) -> Result<bool> {
        OpenAIClient::health_check(self).await
    }

    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn model_id(&self) -> &str {
        self.model().as_str()
    }
}

/// Implement LLMClient trait for AnthropicClient
#[async_trait]
impl LLMClient for AnthropicClient {
    async fn complete(&self, prompt: &str) -> Result<String> {
        AnthropicClient::complete(self, prompt).await
    }

    async fn complete_with_retry(&self, prompt: &str) -> Result<String> {
        AnthropicClient::complete_with_retry(self, prompt).await
    }

    async fn complete_streaming(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        AnthropicClient::complete_streaming(self, prompt).await
    }

    async fn health_check(&self) -> Result<bool> {
        AnthropicClient::health_check(self).await
    }

    fn provider_name(&self) -> &'static str {
        "anthropic"
    }

    fn model_id(&self) -> &str {
        self.model().as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_re_exports() {
        // Verify all re-exports work
        let _ = OpenAIModel::Gpt4o;
        let _ = DEFAULT_OPENAI_ENDPOINT;
        let _ = AnthropicModel::ClaudeSonnet4;
        let _ = DEFAULT_ANTHROPIC_ENDPOINT;
        let _ = DEFAULT_TIMEOUT_SECS;
        let _ = DEFAULT_MAX_RETRIES;
    }

    #[test]
    fn test_openai_client_implements_llm_client() {
        // This test verifies the trait implementation compiles
        fn assert_llm_client<T: LLMClient>() {}
        assert_llm_client::<OpenAIClient>();
    }

    #[test]
    fn test_anthropic_client_implements_llm_client() {
        // This test verifies the trait implementation compiles
        fn assert_llm_client<T: LLMClient>() {}
        assert_llm_client::<AnthropicClient>();
    }

    #[test]
    fn test_message_role() {
        assert_eq!(MessageRole::System.as_str(), "system");
        assert_eq!(MessageRole::User.as_str(), "user");
        assert_eq!(MessageRole::Assistant.as_str(), "assistant");
    }

    #[test]
    fn test_chat_message_constructors() {
        let user_msg = ChatMessage::user("Hello");
        assert_eq!(user_msg.role, MessageRole::User);
        assert_eq!(user_msg.content, "Hello");

        let assistant_msg = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant_msg.role, MessageRole::Assistant);

        let system_msg = ChatMessage::system("You are helpful.");
        assert_eq!(system_msg.role, MessageRole::System);
    }

    #[test]
    fn test_generation_options_presets() {
        let creative = GenerationOptions::creative();
        assert_eq!(creative.temperature, Some(1.0));

        let deterministic = GenerationOptions::deterministic();
        assert_eq!(deterministic.temperature, Some(0.1));

        let code = GenerationOptions::code();
        assert_eq!(code.temperature, Some(0.2));
    }

    #[test]
    fn test_completion_response() {
        let response = CompletionResponse::new("Hello world", "gpt-4o");
        assert_eq!(response.content, "Hello world");
        assert_eq!(response.model, "gpt-4o");
        assert!(response.done);
    }

    #[test]
    fn test_model_info() {
        let info = ModelInfo::new("llama3.2");
        assert_eq!(info.name, "llama3.2");
        assert!(info.supports_chat);
        assert!(!info.supports_embeddings);
    }

    #[test]
    fn test_ollama_client_implements_ollama_llm_client() {
        // This test verifies the trait implementation compiles
        fn assert_ollama_llm_client<T: OllamaLLMClient>() {}
        assert_ollama_llm_client::<OllamaClient>();
    }

    #[test]
    fn test_chat_message_with_name() {
        let msg = ChatMessage::user("test").with_name("alice");
        assert_eq!(msg.name, Some("alice".to_string()));
    }

    #[test]
    fn test_health_check_struct() {
        let check = HealthCheck {
            healthy: true,
            provider: "ollama".to_string(),
            endpoint: "http://localhost:11434".to_string(),
            latency_ms: Some(50),
            error: None,
            model_count: Some(3),
        };
        assert!(check.healthy);
        assert_eq!(check.provider, "ollama");
        assert_eq!(check.model_count, Some(3));
    }
}
