//! Example: LLM Router
//!
//! This example demonstrates the LLM Router for intelligent request
//! routing across multiple providers (OpenAI, Anthropic, Ollama, etc.)
//! with circuit breaker, fallback chains, and cost tracking.
//!
//! Architecture:
//! ```text
//! +------------------+
//! |    LLMRouter     |  <-- Selects appropriate provider
//! +------------------+
//!          |
//!    +-----+-----+-----+-----+
//!    |     |     |     |     |
//!    v     v     v     v     v
//! +------+------+------+------+
//! |OpenAI|Claude|Gemini|Ollama|
//! +------+------+------+------+
//! ```

use panpsychism::{
    llm::{
        // Clients
        AnthropicClientBuilder, AnthropicConfig, AnthropicModel,
        OllamaClientBuilder, OllamaConfig,
        OpenAIClientBuilder, OpenAIConfig, OpenAIModel,
        // Router
        LLMProvider, LLMRouterBuilder, ProviderConfig, RoutingStrategy,
        // Shared types
        ChatMessage, GenerationOptions,
    },
    Result,
};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    panpsychism::setup_logging();

    println!("=== Panpsychism LLM Router Example ===\n");

    // === Part 1: Individual Clients ===
    println!("--- Part 1: Individual Clients ---\n");

    // OpenAI Client
    if let Ok(openai_config) = OpenAIConfig::from_env() {
        let openai = OpenAIClientBuilder::new()
            .config(openai_config)
            .model(OpenAIModel::Gpt4oMini)
            .timeout(Duration::from_secs(30))
            .build()?;

        println!("OpenAI client ready: {:?}", openai.model());
    } else {
        println!("OpenAI: OPENAI_API_KEY not set");
    }

    // Anthropic Client
    if let Ok(anthropic_config) = AnthropicConfig::from_env() {
        let anthropic = AnthropicClientBuilder::new()
            .config(anthropic_config)
            .model(AnthropicModel::Claude3Haiku)
            .build()?;

        println!("Anthropic client ready: {:?}", anthropic.model());
    } else {
        println!("Anthropic: ANTHROPIC_API_KEY not set");
    }

    // Ollama Client (local)
    let ollama_config = OllamaConfig::default();
    let ollama = OllamaClientBuilder::new()
        .config(ollama_config)
        .model("llama3.2")
        .build()?;

    println!("Ollama client configured (local): llama3.2");

    // === Part 2: Router Setup ===
    println!("\n--- Part 2: Router Setup ---\n");

    // Create a router with multiple providers
    let router = LLMRouterBuilder::new()
        .provider(
            ProviderConfig::new(LLMProvider::OpenAI)
                .api_key_env("OPENAI_API_KEY")
                .model("gpt-4o-mini")
                .priority(0) // Primary
                .max_retries(3)
                .timeout(Duration::from_secs(30)),
        )
        .provider(
            ProviderConfig::new(LLMProvider::Anthropic)
                .api_key_env("ANTHROPIC_API_KEY")
                .model("claude-3-haiku-20240307")
                .priority(1) // Fallback 1
                .max_retries(2),
        )
        .provider(
            ProviderConfig::new(LLMProvider::Ollama)
                .endpoint("http://localhost:11434")
                .model("llama3.2")
                .priority(2), // Fallback 2 (local)
        )
        .strategy(RoutingStrategy::Primary)
        .fallback_chain(vec![
            LLMProvider::OpenAI,
            LLMProvider::Anthropic,
            LLMProvider::Ollama,
        ])
        .circuit_breaker_threshold(5) // Trip after 5 failures
        .circuit_breaker_reset(Duration::from_secs(60))
        .build()?;

    println!("Router configured with {} providers", router.provider_count());
    println!("Strategy: {:?}", router.strategy());
    println!("Fallback chain: {:?}", router.fallback_chain());

    // === Part 3: Making Requests ===
    println!("\n--- Part 3: Making Requests ---\n");

    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Explain Rust's ownership model in 3 sentences."),
    ];

    let options = GenerationOptions::default()
        .max_tokens(200)
        .temperature(0.7);

    // Route the request
    println!("Sending request...");
    match router.chat(&messages, Some(&options)).await {
        Ok(response) => {
            println!("Provider used: {:?}", response.provider);
            println!("Model: {}", response.model);
            println!("Response:\n{}", response.content);
            println!("\nUsage:");
            println!("  Prompt tokens: {}", response.usage.prompt_tokens);
            println!("  Completion tokens: {}", response.usage.completion_tokens);
            println!("  Total tokens: {}", response.usage.total_tokens);
        }
        Err(e) => {
            eprintln!("Request failed: {}", e);
            eprintln!("(This is expected if no API keys are configured)");
        }
    }

    // === Part 4: Routing Strategies ===
    println!("\n--- Part 4: Routing Strategies ---\n");

    demonstrate_strategies()?;

    // === Part 5: Cost Tracking ===
    println!("\n--- Part 5: Cost Tracking ---\n");

    let cost_snapshot = router.cost_tracker().snapshot();
    println!("Cost tracking:");
    println!("  Total requests: {}", cost_snapshot.total_requests);
    println!("  Total tokens: {}", cost_snapshot.total_tokens);
    println!("  Estimated cost: ${:.4}", cost_snapshot.estimated_cost);
    println!("  By provider:");
    for (provider, stats) in &cost_snapshot.by_provider {
        println!(
            "    {:?}: {} requests, {} tokens",
            provider, stats.requests, stats.tokens
        );
    }

    // === Part 6: Health Status ===
    println!("\n--- Part 6: Health Status ---\n");

    for provider in LLMProvider::all() {
        let health = router.provider_health(*provider);
        println!(
            "{:?}: {} (errors: {})",
            provider,
            if health.is_healthy { "healthy" } else { "unhealthy" },
            health.error_count
        );
    }

    Ok(())
}

/// Demonstrate different routing strategies
fn demonstrate_strategies() -> Result<()> {
    println!("Available strategies:");
    println!("  - Primary: Use primary provider, fallback on failure");
    println!("  - LoadBalance: Distribute requests across providers");
    println!("  - CostOptimized: Prefer cheaper providers");
    println!("  - QualityOptimized: Prefer best-quality providers");
    println!("  - LatencyOptimized: Prefer fastest providers");
    println!("  - Hybrid: Balance cost, quality, and latency");

    // Example: Create router with different strategy
    let _cost_router = LLMRouterBuilder::new()
        .provider(ProviderConfig::new(LLMProvider::OpenAI).api_key_env("OPENAI_API_KEY"))
        .provider(ProviderConfig::new(LLMProvider::Ollama))
        .strategy(RoutingStrategy::CostOptimized)
        .build()?;

    println!("\nCostOptimized router prefers: Ollama (free) > OpenAI (paid)");

    Ok(())
}

/// Example: Streaming response
#[allow(dead_code)]
async fn streaming_example(router: &panpsychism::llm::LLMRouter) -> Result<()> {
    use futures::StreamExt;

    let messages = vec![ChatMessage::user("Write a short poem about Rust programming.")];

    let mut stream = router.chat_stream(&messages, None).await?;

    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => print!("{}", text),
            Err(e) => eprintln!("\nStream error: {}", e),
        }
    }
    println!();

    Ok(())
}

/// Example: Manual provider selection
#[allow(dead_code)]
async fn manual_provider_selection(router: &panpsychism::llm::LLMRouter) -> Result<()> {
    let messages = vec![ChatMessage::user("Hello!")];

    // Force specific provider
    let response = router
        .chat_with_provider(&messages, LLMProvider::Anthropic, None)
        .await?;

    println!("Forced Anthropic response: {}", response.content);

    Ok(())
}

/// Example: Custom circuit breaker handling
#[allow(dead_code)]
async fn circuit_breaker_demo(router: &panpsychism::llm::LLMRouter) -> Result<()> {
    // Check circuit state
    for provider in LLMProvider::all() {
        let state = router.circuit_state(*provider);
        println!("{:?} circuit: {:?}", provider, state);
    }

    // Manually trip/reset circuit
    router.trip_circuit(LLMProvider::OpenAI);
    println!("OpenAI circuit tripped manually");

    // Wait and reset
    tokio::time::sleep(Duration::from_secs(5)).await;
    router.reset_circuit(LLMProvider::OpenAI);
    println!("OpenAI circuit reset");

    Ok(())
}

/// Example: Batch processing with router
#[allow(dead_code)]
async fn batch_processing(router: &panpsychism::llm::LLMRouter) -> Result<Vec<String>> {
    use futures::future::join_all;

    let queries = vec![
        "What is Rust?",
        "Explain ownership",
        "What are lifetimes?",
    ];

    let futures: Vec<_> = queries
        .iter()
        .map(|q| {
            let messages = vec![ChatMessage::user(*q)];
            router.chat(&messages, None)
        })
        .collect();

    let results = join_all(futures).await;

    results
        .into_iter()
        .map(|r| r.map(|resp| resp.content))
        .collect()
}
