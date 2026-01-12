//! # Ollama LLM Client
//!
//! Local model provider implementation using Ollama.
//!
//! Ollama is a local-first LLM runner that requires no API keys. This module
//! provides a complete client implementation for interacting with Ollama's API.
//!
//! ## Features
//!
//! - **Local-first**: No API keys or cloud dependencies
//! - **Model management**: List, pull, and query model information
//! - **Streaming support**: Token-by-token response streaming
//! - **Chat and completion**: Both conversation and raw completion APIs
//! - **Health checking**: Automatic availability detection
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use panpsychism::llm::{OllamaClient, LLMClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let client = OllamaClient::builder()
//!         .endpoint("http://localhost:11434")
//!         .default_model("llama3.2")
//!         .build()
//!         .await?;
//!
//!     let response = client.complete("What is Rust?").await?;
//!     println!("{}", response.content);
//!     Ok(())
//! }
//! ```
//!
//! ## Ollama API Reference
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/api/tags` | GET | List available models |
//! | `/api/generate` | POST | Generate completion |
//! | `/api/chat` | POST | Chat conversation |
//! | `/api/pull` | POST | Pull/download model |
//! | `/api/show` | POST | Get model info |

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::{
    ChatMessage, ChatResponse, CompletionResponse, GenerationOptions, HealthCheck,
    MessageRole, ModelInfo, OllamaLLMClient,
};
use crate::error::{Error, Result};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default Ollama endpoint.
pub const DEFAULT_ENDPOINT: &str = "http://localhost:11434";

/// Default Ollama model.
pub const DEFAULT_MODEL: &str = "llama3.2";

/// Default timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 300;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for Ollama client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Ollama API endpoint (default: http://localhost:11434).
    pub endpoint: String,
    /// Default model to use for requests.
    pub default_model: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Whether to auto-pull model if not available.
    pub pull_if_missing: bool,
    /// Keep-alive duration for loaded models (e.g., "5m", "1h").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_ENDPOINT.to_string(),
            default_model: DEFAULT_MODEL.to_string(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            pull_if_missing: false,
            keep_alive: None,
        }
    }
}

impl OllamaConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Set the default model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Enable auto-pull for missing models.
    pub fn with_auto_pull(mut self, enabled: bool) -> Self {
        self.pull_if_missing = enabled;
        self
    }
}

// ============================================================================
// OLLAMA MODEL INFO
// ============================================================================

/// Information about an Ollama model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    /// Model name (e.g., "llama3.2:latest").
    pub name: String,
    /// Model size in bytes.
    #[serde(default)]
    pub size: u64,
    /// Model digest (SHA256).
    #[serde(default)]
    pub digest: String,
    /// Last modification timestamp.
    #[serde(default)]
    pub modified_at: String,
    /// Model parameters (e.g., "8B").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<String>,
    /// Quantization level (e.g., "Q4_0").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Model family.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Model format (e.g., "gguf").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

impl OllamaModel {
    /// Get the base model name without tag.
    pub fn base_name(&self) -> &str {
        self.name.split(':').next().unwrap_or(&self.name)
    }

    /// Get the tag (e.g., "latest", "7b-q4").
    pub fn tag(&self) -> &str {
        self.name.split(':').nth(1).unwrap_or("latest")
    }

    /// Get size in human-readable format.
    pub fn size_human(&self) -> String {
        const GB: u64 = 1_000_000_000;
        const MB: u64 = 1_000_000;
        const KB: u64 = 1_000;

        if self.size >= GB {
            format!("{:.1} GB", self.size as f64 / GB as f64)
        } else if self.size >= MB {
            format!("{:.1} MB", self.size as f64 / MB as f64)
        } else if self.size >= KB {
            format!("{:.1} KB", self.size as f64 / KB as f64)
        } else {
            format!("{} B", self.size)
        }
    }
}

impl From<OllamaModel> for ModelInfo {
    fn from(model: OllamaModel) -> Self {
        ModelInfo {
            name: model.name,
            size: Some(model.size),
            family: model.family,
            parameters: model.parameters,
            quantization: model.quantization,
            context_length: None,
            supports_chat: true,
            supports_embeddings: false,
        }
    }
}

// ============================================================================
// OLLAMA API REQUEST/RESPONSE TYPES
// ============================================================================

/// Ollama generate request body.
#[derive(Debug, Clone, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaGenerateOptions>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

/// Ollama generate response body.
#[derive(Debug, Clone, Deserialize)]
struct OllamaGenerateResponse {
    model: String,
    #[serde(default)]
    response: String,
    done: bool,
    #[serde(default)]
    context: Vec<i64>,
    #[serde(default)]
    total_duration: u64,
    #[serde(default)]
    load_duration: u64,
    #[serde(default)]
    prompt_eval_count: u32,
    #[serde(default)]
    prompt_eval_duration: u64,
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    eval_duration: u64,
}

/// Ollama chat message for API.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct OllamaChatMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: String,
}

impl From<&ChatMessage> for OllamaChatMessage {
    fn from(msg: &ChatMessage) -> Self {
        Self {
            role: msg.role.as_str().to_string(),
            content: msg.content.clone(),
        }
    }
}

impl From<OllamaChatMessage> for ChatMessage {
    fn from(msg: OllamaChatMessage) -> Self {
        let role = match msg.role.as_str() {
            "system" => MessageRole::System,
            "assistant" => MessageRole::Assistant,
            _ => MessageRole::User,
        };
        ChatMessage {
            role,
            content: msg.content,
            name: None,
        }
    }
}

/// Ollama chat request body.
#[derive(Debug, Clone, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaGenerateOptions>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

/// Ollama chat response body.
#[derive(Debug, Clone, Deserialize)]
struct OllamaChatResponse {
    model: String,
    #[serde(default)]
    message: OllamaChatMessage,
    done: bool,
    #[serde(default)]
    total_duration: u64,
    #[serde(default)]
    load_duration: u64,
    #[serde(default)]
    prompt_eval_count: u32,
    #[serde(default)]
    prompt_eval_duration: u64,
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    eval_duration: u64,
}

/// Ollama generation options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaGenerateOptions {
    /// Temperature for sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Number of tokens to predict.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    /// Random seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Repeat penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    /// Context window size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    /// Mirostat mode (0, 1, or 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<u32>,
    /// Mirostat tau.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,
    /// Mirostat eta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,
}

impl From<&GenerationOptions> for OllamaGenerateOptions {
    fn from(opts: &GenerationOptions) -> Self {
        Self {
            temperature: opts.temperature,
            top_p: opts.top_p,
            top_k: opts.top_k,
            num_predict: opts.max_tokens,
            seed: opts.seed,
            stop: opts.stop.clone(),
            repeat_penalty: opts.frequency_penalty.map(|f| 1.0 + f.abs()),
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
        }
    }
}

/// Ollama model list response.
#[derive(Debug, Clone, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelEntry>,
}

/// Entry in model list.
#[derive(Debug, Clone, Deserialize)]
struct OllamaModelEntry {
    name: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    digest: String,
    #[serde(default)]
    modified_at: String,
    #[serde(default)]
    details: OllamaModelDetails,
}

/// Model details from list.
#[derive(Debug, Clone, Deserialize, Default)]
struct OllamaModelDetails {
    #[serde(default)]
    family: String,
    #[serde(default)]
    parameter_size: String,
    #[serde(default)]
    quantization_level: String,
    #[serde(default)]
    format: String,
}

impl From<OllamaModelEntry> for OllamaModel {
    fn from(entry: OllamaModelEntry) -> Self {
        Self {
            name: entry.name,
            size: entry.size,
            digest: entry.digest,
            modified_at: entry.modified_at,
            parameters: if entry.details.parameter_size.is_empty() {
                None
            } else {
                Some(entry.details.parameter_size)
            },
            quantization: if entry.details.quantization_level.is_empty() {
                None
            } else {
                Some(entry.details.quantization_level)
            },
            family: if entry.details.family.is_empty() {
                None
            } else {
                Some(entry.details.family)
            },
            format: if entry.details.format.is_empty() {
                None
            } else {
                Some(entry.details.format)
            },
        }
    }
}

/// Ollama show request.
#[derive(Debug, Clone, Serialize)]
struct OllamaShowRequest {
    name: String,
}

/// Ollama show response.
#[derive(Debug, Clone, Deserialize)]
struct OllamaShowResponse {
    #[serde(default)]
    modelfile: String,
    #[serde(default)]
    parameters: String,
    #[serde(default)]
    template: String,
    #[serde(default)]
    details: OllamaModelDetails,
}

/// Ollama pull request.
#[derive(Debug, Clone, Serialize)]
struct OllamaPullRequest {
    name: String,
    stream: bool,
}

/// Ollama pull response (streaming).
#[derive(Debug, Clone, Deserialize)]
struct OllamaPullResponse {
    #[serde(default)]
    status: String,
    #[serde(default)]
    digest: String,
    #[serde(default)]
    total: u64,
    #[serde(default)]
    completed: u64,
}

// ============================================================================
// OLLAMA CLIENT
// ============================================================================

/// Ollama LLM client.
///
/// Provides a complete implementation for interacting with Ollama's local API.
/// Supports both completion and chat endpoints, model management, and health checks.
pub struct OllamaClient {
    /// Ollama API endpoint.
    endpoint: String,
    /// Default model to use.
    default_model: String,
    /// HTTP client for API requests.
    http_client: Client,
    /// Cached list of available models.
    available_models: Vec<OllamaModel>,
    /// Configuration.
    config: OllamaConfig,
}

impl OllamaClient {
    /// Create a new OllamaClient with default configuration.
    ///
    /// # Returns
    ///
    /// A new `OllamaClient` or an error if connection fails.
    pub async fn new() -> Result<Self> {
        Self::with_config(OllamaConfig::default()).await
    }

    /// Create a new OllamaClient with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Client configuration
    ///
    /// # Returns
    ///
    /// A new `OllamaClient` or an error if connection fails.
    pub async fn with_config(config: OllamaConfig) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| Error::internal(format!("Failed to create HTTP client: {}", e)))?;

        let mut client = Self {
            endpoint: config.endpoint.clone(),
            default_model: config.default_model.clone(),
            http_client,
            available_models: Vec::new(),
            config,
        };

        // Try to refresh models list (don't fail if Ollama is not running)
        let _ = client.refresh_models().await;

        Ok(client)
    }

    /// Create a new builder for OllamaClient.
    pub fn builder() -> OllamaClientBuilder {
        OllamaClientBuilder::new()
    }

    /// Get the endpoint URL.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Get the list of cached available models.
    pub fn available_models(&self) -> &[OllamaModel] {
        &self.available_models
    }

    /// Refresh the list of available models from Ollama.
    pub async fn refresh_models(&mut self) -> Result<Vec<OllamaModel>> {
        let url = format!("{}/api/tags", self.endpoint);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::api_connection_failed_with_source(&self.endpoint, e))?;

        if !response.status().is_success() {
            return Err(Error::api_response(
                response.status().as_u16(),
                "Failed to list models",
            ));
        }

        let tags: OllamaTagsResponse = response.json().await.map_err(|e| {
            Error::internal(format!("Failed to parse models response: {}", e))
        })?;

        self.available_models = tags.models.into_iter().map(OllamaModel::from).collect();
        Ok(self.available_models.clone())
    }

    /// Check if a model is available locally.
    pub fn has_model(&self, name: &str) -> bool {
        self.available_models
            .iter()
            .any(|m| m.name == name || m.base_name() == name)
    }

    /// Pull a model from Ollama registry.
    ///
    /// # Arguments
    ///
    /// * `name` - Model name to pull
    ///
    /// # Returns
    ///
    /// The pulled model information.
    pub async fn pull_model(&self, name: &str) -> Result<OllamaModel> {
        let url = format!("{}/api/pull", self.endpoint);

        let request = OllamaPullRequest {
            name: name.to_string(),
            stream: false,
        };

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::api_connection_failed_with_source(&self.endpoint, e))?;

        if !response.status().is_success() {
            return Err(Error::api_response(
                response.status().as_u16(),
                format!("Failed to pull model: {}", name),
            ));
        }

        // Wait for pull to complete
        let _pull: OllamaPullResponse = response.json().await.map_err(|e| {
            Error::internal(format!("Failed to parse pull response: {}", e))
        })?;

        // Get model info
        self.show_model(name).await
    }

    /// Get detailed information about a model.
    ///
    /// # Arguments
    ///
    /// * `name` - Model name
    ///
    /// # Returns
    ///
    /// Detailed model information.
    pub async fn show_model(&self, name: &str) -> Result<OllamaModel> {
        let url = format!("{}/api/show", self.endpoint);

        let request = OllamaShowRequest {
            name: name.to_string(),
        };

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::api_connection_failed_with_source(&self.endpoint, e))?;

        if !response.status().is_success() {
            return Err(Error::api_response(
                response.status().as_u16(),
                format!("Model not found: {}", name),
            ));
        }

        let show: OllamaShowResponse = response.json().await.map_err(|e| {
            Error::internal(format!("Failed to parse show response: {}", e))
        })?;

        Ok(OllamaModel {
            name: name.to_string(),
            size: 0,
            digest: String::new(),
            modified_at: String::new(),
            parameters: if show.details.parameter_size.is_empty() {
                None
            } else {
                Some(show.details.parameter_size)
            },
            quantization: if show.details.quantization_level.is_empty() {
                None
            } else {
                Some(show.details.quantization_level)
            },
            family: if show.details.family.is_empty() {
                None
            } else {
                Some(show.details.family)
            },
            format: if show.details.format.is_empty() {
                None
            } else {
                Some(show.details.format)
            },
        })
    }

    /// Generate a completion (internal implementation).
    async fn generate_internal(
        &self,
        prompt: &str,
        model: &str,
        system: Option<&str>,
        options: Option<&GenerationOptions>,
    ) -> Result<CompletionResponse> {
        let url = format!("{}/api/generate", self.endpoint);
        let start = Instant::now();

        let request = OllamaGenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            system: system.map(String::from),
            options: options.map(OllamaGenerateOptions::from),
            stream: false,
            keep_alive: self.config.keep_alive.clone(),
        };

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::api_connection_failed_with_source(&self.endpoint, e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::api_response(status, body));
        }

        let gen: OllamaGenerateResponse = response.json().await.map_err(|e| {
            Error::internal(format!("Failed to parse generate response: {}", e))
        })?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(CompletionResponse {
            content: gen.response,
            model: gen.model,
            done: gen.done,
            prompt_tokens: Some(gen.prompt_eval_count),
            completion_tokens: Some(gen.eval_count),
            total_tokens: Some(gen.prompt_eval_count + gen.eval_count),
            duration_ms: Some(duration_ms),
            metadata: Default::default(),
        })
    }

    /// Chat with the model (internal implementation).
    async fn chat_internal(
        &self,
        messages: &[ChatMessage],
        model: &str,
        options: Option<&GenerationOptions>,
    ) -> Result<ChatResponse> {
        let url = format!("{}/api/chat", self.endpoint);
        let start = Instant::now();

        let ollama_messages: Vec<OllamaChatMessage> =
            messages.iter().map(OllamaChatMessage::from).collect();

        let request = OllamaChatRequest {
            model: model.to_string(),
            messages: ollama_messages,
            options: options.map(OllamaGenerateOptions::from),
            stream: false,
            keep_alive: self.config.keep_alive.clone(),
        };

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::api_connection_failed_with_source(&self.endpoint, e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::api_response(status, body));
        }

        let chat: OllamaChatResponse = response.json().await.map_err(|e| {
            Error::internal(format!("Failed to parse chat response: {}", e))
        })?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ChatResponse {
            message: chat.message.into(),
            model: chat.model,
            done: chat.done,
            prompt_tokens: Some(chat.prompt_eval_count),
            completion_tokens: Some(chat.eval_count),
            total_tokens: Some(chat.prompt_eval_count + chat.eval_count),
            duration_ms: Some(duration_ms),
        })
    }
}

// ============================================================================
// OLLAMA LLM CLIENT TRAIT IMPLEMENTATION
// ============================================================================

#[async_trait]
impl OllamaLLMClient for OllamaClient {
    fn provider(&self) -> &'static str {
        "ollama"
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    async fn complete(&self, prompt: &str) -> Result<CompletionResponse> {
        self.generate_internal(prompt, &self.default_model, None, None)
            .await
    }

    async fn complete_with_options(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<CompletionResponse> {
        self.generate_internal(prompt, &self.default_model, None, Some(options))
            .await
    }

    async fn complete_with_model(
        &self,
        prompt: &str,
        model: &str,
        options: &GenerationOptions,
    ) -> Result<CompletionResponse> {
        self.generate_internal(prompt, model, None, Some(options))
            .await
    }

    async fn chat(&self, messages: &[ChatMessage]) -> Result<ChatResponse> {
        self.chat_internal(messages, &self.default_model, None)
            .await
    }

    async fn chat_with_options(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
    ) -> Result<ChatResponse> {
        self.chat_internal(messages, &self.default_model, Some(options))
            .await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.endpoint);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::api_connection_failed_with_source(&self.endpoint, e))?;

        if !response.status().is_success() {
            return Err(Error::api_response(
                response.status().as_u16(),
                "Failed to list models",
            ));
        }

        let tags: OllamaTagsResponse = response.json().await.map_err(|e| {
            Error::internal(format!("Failed to parse models response: {}", e))
        })?;

        Ok(tags
            .models
            .into_iter()
            .map(OllamaModel::from)
            .map(ModelInfo::from)
            .collect())
    }

    async fn ollama_health_check(&self) -> Result<HealthCheck> {
        let url = format!("{}/api/tags", self.endpoint);
        let start = Instant::now();

        match self.http_client.get(&url).send().await {
            Ok(response) => {
                let latency = start.elapsed().as_millis() as u64;
                let healthy = response.status().is_success();

                let model_count = if healthy {
                    response
                        .json::<OllamaTagsResponse>()
                        .await
                        .map(|t| t.models.len())
                        .ok()
                } else {
                    None
                };

                Ok(HealthCheck {
                    healthy,
                    provider: "ollama".to_string(),
                    endpoint: self.endpoint.clone(),
                    latency_ms: Some(latency),
                    error: if healthy {
                        None
                    } else {
                        Some("Unhealthy response".to_string())
                    },
                    model_count,
                })
            }
            Err(e) => Ok(HealthCheck {
                healthy: false,
                provider: "ollama".to_string(),
                endpoint: self.endpoint.clone(),
                latency_ms: None,
                error: Some(e.to_string()),
                model_count: None,
            }),
        }
    }

    async fn model_info(&self, model: &str) -> Result<ModelInfo> {
        let ollama_model = self.show_model(model).await?;
        Ok(ollama_model.into())
    }
}

// ============================================================================
// BUILDER
// ============================================================================

/// Builder for OllamaClient.
///
/// Provides a fluent API for constructing an OllamaClient with custom configuration.
///
/// # Example
///
/// ```rust,ignore
/// let client = OllamaClient::builder()
///     .endpoint("http://localhost:11434")
///     .default_model("llama3.2")
///     .timeout_secs(600)
///     .pull_if_missing(true)
///     .build()
///     .await?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct OllamaClientBuilder {
    config: OllamaConfig,
}

impl OllamaClientBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: OllamaConfig::default(),
        }
    }

    /// Set the Ollama API endpoint.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.endpoint = endpoint.into();
        self
    }

    /// Set the default model.
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    /// Set the request timeout in seconds.
    pub fn timeout_secs(mut self, timeout: u64) -> Self {
        self.config.timeout_secs = timeout;
        self
    }

    /// Enable auto-pulling of missing models.
    pub fn pull_if_missing(mut self, enabled: bool) -> Self {
        self.config.pull_if_missing = enabled;
        self
    }

    /// Set the keep-alive duration for loaded models.
    pub fn keep_alive(mut self, duration: impl Into<String>) -> Self {
        self.config.keep_alive = Some(duration.into());
        self
    }

    /// Build the OllamaClient.
    pub async fn build(self) -> Result<OllamaClient> {
        OllamaClient::with_config(self.config).await
    }

    /// Get the current configuration.
    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Configuration Tests
    // ========================================================================

    #[test]
    fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(config.default_model, DEFAULT_MODEL);
        assert_eq!(config.timeout_secs, DEFAULT_TIMEOUT_SECS);
        assert!(!config.pull_if_missing);
        assert!(config.keep_alive.is_none());
    }

    #[test]
    fn test_ollama_config_builder_pattern() {
        let config = OllamaConfig::new()
            .with_endpoint("http://custom:8080")
            .with_model("mistral")
            .with_timeout(60)
            .with_auto_pull(true);

        assert_eq!(config.endpoint, "http://custom:8080");
        assert_eq!(config.default_model, "mistral");
        assert_eq!(config.timeout_secs, 60);
        assert!(config.pull_if_missing);
    }

    #[test]
    fn test_ollama_config_serde() {
        let config = OllamaConfig::new()
            .with_endpoint("http://test:11434")
            .with_model("phi3");

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("http://test:11434"));
        assert!(json.contains("phi3"));

        let parsed: OllamaConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.endpoint, "http://test:11434");
        assert_eq!(parsed.default_model, "phi3");
    }

    // ========================================================================
    // OllamaModel Tests
    // ========================================================================

    #[test]
    fn test_ollama_model_base_name() {
        let model = OllamaModel {
            name: "llama3.2:latest".to_string(),
            size: 4_000_000_000,
            digest: "abc123".to_string(),
            modified_at: "2024-01-01".to_string(),
            parameters: Some("8B".to_string()),
            quantization: Some("Q4_0".to_string()),
            family: Some("llama".to_string()),
            format: Some("gguf".to_string()),
        };

        assert_eq!(model.base_name(), "llama3.2");
        assert_eq!(model.tag(), "latest");
    }

    #[test]
    fn test_ollama_model_no_tag() {
        let model = OllamaModel {
            name: "mistral".to_string(),
            size: 0,
            digest: String::new(),
            modified_at: String::new(),
            parameters: None,
            quantization: None,
            family: None,
            format: None,
        };

        assert_eq!(model.base_name(), "mistral");
        assert_eq!(model.tag(), "latest");
    }

    #[test]
    fn test_ollama_model_size_human() {
        let mut model = OllamaModel {
            name: "test".to_string(),
            size: 0,
            digest: String::new(),
            modified_at: String::new(),
            parameters: None,
            quantization: None,
            family: None,
            format: None,
        };

        model.size = 500;
        assert_eq!(model.size_human(), "500 B");

        model.size = 1_500;
        assert!(model.size_human().contains("KB"));

        model.size = 1_500_000;
        assert!(model.size_human().contains("MB"));

        model.size = 4_500_000_000;
        assert!(model.size_human().contains("GB"));
    }

    #[test]
    fn test_ollama_model_to_model_info() {
        let ollama = OllamaModel {
            name: "llama3.2:latest".to_string(),
            size: 4_000_000_000,
            digest: "abc123".to_string(),
            modified_at: "2024-01-01".to_string(),
            parameters: Some("8B".to_string()),
            quantization: Some("Q4_0".to_string()),
            family: Some("llama".to_string()),
            format: Some("gguf".to_string()),
        };

        let info: ModelInfo = ollama.into();
        assert_eq!(info.name, "llama3.2:latest");
        assert_eq!(info.size, Some(4_000_000_000));
        assert_eq!(info.family, Some("llama".to_string()));
        assert_eq!(info.parameters, Some("8B".to_string()));
        assert_eq!(info.quantization, Some("Q4_0".to_string()));
        assert!(info.supports_chat);
    }

    // ========================================================================
    // Request/Response Type Tests
    // ========================================================================

    #[test]
    fn test_ollama_chat_message_from() {
        let chat_msg = ChatMessage::user("Hello");
        let ollama_msg: OllamaChatMessage = (&chat_msg).into();

        assert_eq!(ollama_msg.role, "user");
        assert_eq!(ollama_msg.content, "Hello");
    }

    #[test]
    fn test_ollama_chat_message_to() {
        let ollama_msg = OllamaChatMessage {
            role: "assistant".to_string(),
            content: "Hi there".to_string(),
        };

        let chat_msg: ChatMessage = ollama_msg.into();
        assert_eq!(chat_msg.role, MessageRole::Assistant);
        assert_eq!(chat_msg.content, "Hi there");
    }

    #[test]
    fn test_ollama_generate_options_from() {
        let opts = GenerationOptions {
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: Some(50),
            max_tokens: Some(1000),
            stop: Some(vec!["END".to_string()]),
            seed: Some(42),
            frequency_penalty: Some(0.5),
            presence_penalty: None,
        };

        let ollama_opts: OllamaGenerateOptions = (&opts).into();
        assert_eq!(ollama_opts.temperature, Some(0.8));
        assert_eq!(ollama_opts.top_p, Some(0.95));
        assert_eq!(ollama_opts.top_k, Some(50));
        assert_eq!(ollama_opts.num_predict, Some(1000));
        assert_eq!(ollama_opts.seed, Some(42));
        assert!(ollama_opts.repeat_penalty.is_some());
    }

    #[test]
    fn test_ollama_generate_options_default() {
        let opts = OllamaGenerateOptions::default();
        assert!(opts.temperature.is_none());
        assert!(opts.top_p.is_none());
        assert!(opts.num_predict.is_none());
    }

    #[test]
    fn test_ollama_model_entry_to_model() {
        let entry = OllamaModelEntry {
            name: "llama3.2:7b-q4".to_string(),
            size: 3_800_000_000,
            digest: "sha256:abc".to_string(),
            modified_at: "2024-01-01T00:00:00Z".to_string(),
            details: OllamaModelDetails {
                family: "llama".to_string(),
                parameter_size: "7B".to_string(),
                quantization_level: "Q4_0".to_string(),
                format: "gguf".to_string(),
            },
        };

        let model: OllamaModel = entry.into();
        assert_eq!(model.name, "llama3.2:7b-q4");
        assert_eq!(model.size, 3_800_000_000);
        assert_eq!(model.family, Some("llama".to_string()));
        assert_eq!(model.parameters, Some("7B".to_string()));
        assert_eq!(model.quantization, Some("Q4_0".to_string()));
    }

    #[test]
    fn test_ollama_model_entry_empty_details() {
        let entry = OllamaModelEntry {
            name: "custom-model".to_string(),
            size: 1_000_000,
            digest: "".to_string(),
            modified_at: "".to_string(),
            details: OllamaModelDetails::default(),
        };

        let model: OllamaModel = entry.into();
        assert_eq!(model.name, "custom-model");
        assert!(model.family.is_none());
        assert!(model.parameters.is_none());
        assert!(model.quantization.is_none());
    }

    // ========================================================================
    // Builder Tests
    // ========================================================================

    #[test]
    fn test_builder_default() {
        let builder = OllamaClientBuilder::new();
        let config = builder.config();

        assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(config.default_model, DEFAULT_MODEL);
    }

    #[test]
    fn test_builder_fluent() {
        let builder = OllamaClientBuilder::new()
            .endpoint("http://custom:8080")
            .default_model("phi3")
            .timeout_secs(120)
            .pull_if_missing(true)
            .keep_alive("5m");

        let config = builder.config();
        assert_eq!(config.endpoint, "http://custom:8080");
        assert_eq!(config.default_model, "phi3");
        assert_eq!(config.timeout_secs, 120);
        assert!(config.pull_if_missing);
        assert_eq!(config.keep_alive, Some("5m".to_string()));
    }

    #[test]
    fn test_builder_clone() {
        let builder = OllamaClientBuilder::new()
            .endpoint("http://test:11434")
            .default_model("llama3");

        let cloned = builder.clone();
        assert_eq!(cloned.config().endpoint, "http://test:11434");
        assert_eq!(cloned.config().default_model, "llama3");
    }

    // ========================================================================
    // Generate Request Serialization Tests
    // ========================================================================

    #[test]
    fn test_generate_request_serialization() {
        let request = OllamaGenerateRequest {
            model: "llama3.2".to_string(),
            prompt: "Hello world".to_string(),
            system: Some("You are helpful".to_string()),
            options: Some(OllamaGenerateOptions {
                temperature: Some(0.7),
                ..Default::default()
            }),
            stream: false,
            keep_alive: Some("5m".to_string()),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"llama3.2\""));
        assert!(json.contains("\"prompt\":\"Hello world\""));
        assert!(json.contains("\"system\":\"You are helpful\""));
        assert!(json.contains("\"stream\":false"));
    }

    #[test]
    fn test_generate_request_minimal() {
        let request = OllamaGenerateRequest {
            model: "llama3.2".to_string(),
            prompt: "Test".to_string(),
            system: None,
            options: None,
            stream: false,
            keep_alive: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("system"));
        assert!(!json.contains("options"));
        assert!(!json.contains("keep_alive"));
    }

    // ========================================================================
    // Generate Response Deserialization Tests
    // ========================================================================

    #[test]
    fn test_generate_response_parsing() {
        let json = r#"{
            "model": "llama3.2",
            "response": "Hello!",
            "done": true,
            "context": [1, 2, 3],
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 200000000,
            "eval_count": 5,
            "eval_duration": 300000000
        }"#;

        let response: OllamaGenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.model, "llama3.2");
        assert_eq!(response.response, "Hello!");
        assert!(response.done);
        assert_eq!(response.prompt_eval_count, 10);
        assert_eq!(response.eval_count, 5);
    }

    #[test]
    fn test_generate_response_minimal() {
        let json = r#"{
            "model": "test",
            "response": "",
            "done": false
        }"#;

        let response: OllamaGenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.model, "test");
        assert!(!response.done);
        assert_eq!(response.context.len(), 0);
        assert_eq!(response.total_duration, 0);
    }

    // ========================================================================
    // Chat Request/Response Tests
    // ========================================================================

    #[test]
    fn test_chat_request_serialization() {
        let request = OllamaChatRequest {
            model: "llama3.2".to_string(),
            messages: vec![
                OllamaChatMessage {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                },
                OllamaChatMessage {
                    role: "assistant".to_string(),
                    content: "Hi!".to_string(),
                },
            ],
            options: None,
            stream: false,
            keep_alive: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"messages\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"role\":\"assistant\""));
    }

    #[test]
    fn test_chat_response_parsing() {
        let json = r#"{
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "Hello there!"
            },
            "done": true,
            "total_duration": 500000000,
            "prompt_eval_count": 5,
            "eval_count": 3
        }"#;

        let response: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.model, "llama3.2");
        assert_eq!(response.message.role, "assistant");
        assert_eq!(response.message.content, "Hello there!");
        assert!(response.done);
    }

    // ========================================================================
    // Tags Response Tests
    // ========================================================================

    #[test]
    fn test_tags_response_parsing() {
        let json = r#"{
            "models": [
                {
                    "name": "llama3.2:latest",
                    "size": 4000000000,
                    "digest": "sha256:abc123",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "details": {
                        "family": "llama",
                        "parameter_size": "8B",
                        "quantization_level": "Q4_0",
                        "format": "gguf"
                    }
                },
                {
                    "name": "mistral:latest",
                    "size": 3500000000,
                    "digest": "sha256:def456",
                    "modified_at": "2024-01-02T00:00:00Z",
                    "details": {}
                }
            ]
        }"#;

        let response: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.models.len(), 2);
        assert_eq!(response.models[0].name, "llama3.2:latest");
        assert_eq!(response.models[1].name, "mistral:latest");
    }

    #[test]
    fn test_tags_response_empty() {
        let json = r#"{"models": []}"#;

        let response: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        assert!(response.models.is_empty());
    }

    // ========================================================================
    // Show Response Tests
    // ========================================================================

    #[test]
    fn test_show_response_parsing() {
        let json = r#"{
            "modelfile": "FROM llama3.2",
            "parameters": "temperature 0.7",
            "template": "{{ .Prompt }}",
            "details": {
                "family": "llama",
                "parameter_size": "8B",
                "quantization_level": "Q4_0",
                "format": "gguf"
            }
        }"#;

        let response: OllamaShowResponse = serde_json::from_str(json).unwrap();
        assert!(response.modelfile.contains("llama3.2"));
        assert_eq!(response.details.family, "llama");
        assert_eq!(response.details.parameter_size, "8B");
    }

    // ========================================================================
    // Pull Response Tests
    // ========================================================================

    #[test]
    fn test_pull_response_parsing() {
        let json = r#"{
            "status": "success",
            "digest": "sha256:abc123",
            "total": 4000000000,
            "completed": 4000000000
        }"#;

        let response: OllamaPullResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "success");
        assert_eq!(response.total, 4_000_000_000);
        assert_eq!(response.completed, 4_000_000_000);
    }

    #[test]
    fn test_pull_response_in_progress() {
        let json = r#"{
            "status": "downloading",
            "digest": "sha256:abc123",
            "total": 4000000000,
            "completed": 2000000000
        }"#;

        let response: OllamaPullResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "downloading");
        assert!(response.completed < response.total);
    }

    // ========================================================================
    // Constants Tests
    // ========================================================================

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_ENDPOINT, "http://localhost:11434");
        assert_eq!(DEFAULT_MODEL, "llama3.2");
        assert_eq!(DEFAULT_TIMEOUT_SECS, 300);
    }

    // ========================================================================
    // Chat Message Role Conversion Tests
    // ========================================================================

    #[test]
    fn test_role_conversion_system() {
        let ollama_msg = OllamaChatMessage {
            role: "system".to_string(),
            content: "You are helpful".to_string(),
        };
        let chat_msg: ChatMessage = ollama_msg.into();
        assert_eq!(chat_msg.role, MessageRole::System);
    }

    #[test]
    fn test_role_conversion_unknown() {
        let ollama_msg = OllamaChatMessage {
            role: "unknown".to_string(),
            content: "Test".to_string(),
        };
        let chat_msg: ChatMessage = ollama_msg.into();
        // Unknown roles default to User
        assert_eq!(chat_msg.role, MessageRole::User);
    }

    // ========================================================================
    // Options Conversion Edge Cases
    // ========================================================================

    #[test]
    fn test_options_negative_frequency_penalty() {
        let opts = GenerationOptions {
            frequency_penalty: Some(-0.5),
            ..Default::default()
        };

        let ollama_opts: OllamaGenerateOptions = (&opts).into();
        // Should convert to positive repeat_penalty
        assert!(ollama_opts.repeat_penalty.unwrap() > 1.0);
    }

    #[test]
    fn test_options_all_none() {
        let opts = GenerationOptions {
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stop: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
        };

        let ollama_opts: OllamaGenerateOptions = (&opts).into();
        assert!(ollama_opts.temperature.is_none());
        assert!(ollama_opts.top_p.is_none());
        assert!(ollama_opts.num_predict.is_none());
        assert!(ollama_opts.repeat_penalty.is_none());
    }

    // ========================================================================
    // Model Details Default Tests
    // ========================================================================

    #[test]
    fn test_model_details_default() {
        let details = OllamaModelDetails::default();
        assert!(details.family.is_empty());
        assert!(details.parameter_size.is_empty());
        assert!(details.quantization_level.is_empty());
        assert!(details.format.is_empty());
    }

    // ========================================================================
    // Serde Skip Serializing Tests
    // ========================================================================

    #[test]
    fn test_ollama_model_serde_skip_none() {
        let model = OllamaModel {
            name: "test".to_string(),
            size: 0,
            digest: "".to_string(),
            modified_at: "".to_string(),
            parameters: None,
            quantization: None,
            family: None,
            format: None,
        };

        let json = serde_json::to_string(&model).unwrap();
        assert!(!json.contains("parameters"));
        assert!(!json.contains("quantization"));
        assert!(!json.contains("family"));
        assert!(!json.contains("format"));
    }

    #[test]
    fn test_ollama_model_serde_with_all() {
        let model = OllamaModel {
            name: "test".to_string(),
            size: 1000,
            digest: "sha256:test".to_string(),
            modified_at: "2024-01-01".to_string(),
            parameters: Some("7B".to_string()),
            quantization: Some("Q4".to_string()),
            family: Some("llama".to_string()),
            format: Some("gguf".to_string()),
        };

        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("\"parameters\":\"7B\""));
        assert!(json.contains("\"quantization\":\"Q4\""));
        assert!(json.contains("\"family\":\"llama\""));
        assert!(json.contains("\"format\":\"gguf\""));
    }

    // ========================================================================
    // Health Check Tests
    // ========================================================================

    #[test]
    fn test_health_check_struct_healthy() {
        let health = HealthCheck {
            healthy: true,
            provider: "ollama".to_string(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            latency_ms: Some(50),
            error: None,
            model_count: Some(3),
        };

        assert!(health.healthy);
        assert!(health.error.is_none());
        assert_eq!(health.model_count, Some(3));
    }

    #[test]
    fn test_health_check_struct_unhealthy() {
        let health = HealthCheck {
            healthy: false,
            provider: "ollama".to_string(),
            endpoint: "http://invalid:11434".to_string(),
            latency_ms: None,
            error: Some("Connection refused".to_string()),
            model_count: None,
        };

        assert!(!health.healthy);
        assert!(health.error.is_some());
        assert!(health.latency_ms.is_none());
    }
}
