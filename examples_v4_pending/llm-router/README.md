# LLM Router Example

Demonstrates intelligent request routing across multiple LLM providers.

## What This Example Does

1. Configure individual LLM clients (OpenAI, Anthropic, Ollama)
2. Set up a router with multiple providers
3. Make requests with automatic failover
4. Explore different routing strategies
5. Track costs and monitor health

## Architecture

```
+------------------+
|    LLMRouter     |  <-- Selects appropriate provider
+------------------+
         |
   +-----+-----+-----+-----+
   |     |     |     |     |
   v     v     v     v     v
+------+------+------+------+
|OpenAI|Claude|Gemini|Ollama|
+------+------+------+------+
```

## Running

```bash
# Set API keys (use at least one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Start local Ollama
ollama serve
ollama pull llama3.2

# Run the example
cargo run --example llm-router
```

## Expected Output

```
=== Panpsychism LLM Router Example ===

--- Part 1: Individual Clients ---

OpenAI client ready: gpt-4o-mini
Anthropic client ready: claude-3-haiku
Ollama client ready: llama3.2

--- Part 2: Router Setup ---

Router configured with 3 providers
Strategy: Primary
Fallback chain: [OpenAI, Anthropic, Ollama]

--- Part 3: Making Requests ---

Sending request...
Provider used: OpenAI
Model: gpt-4o-mini
Response:
Rust's ownership model ensures memory safety without garbage collection...

Usage:
  Prompt tokens: 45
  Completion tokens: 87
  Total tokens: 132

--- Part 5: Cost Tracking ---

Cost tracking:
  Total requests: 1
  Total tokens: 132
  Estimated cost: $0.0001
```

## Key Concepts

### LLM Providers

```rust
// OpenAI
let openai = OpenAIClientBuilder::new()
    .with_config(OpenAIConfig::from_env()?)
    .with_model(OpenAIModel::Gpt4oMini)
    .build()?;

// Anthropic
let anthropic = AnthropicClientBuilder::new()
    .with_config(AnthropicConfig::from_env()?)
    .with_model(AnthropicModel::Claude3Haiku)
    .build()?;

// Ollama (local)
let ollama = OllamaClientBuilder::new()
    .with_model(OllamaModel::Llama3_2)
    .build()?;
```

### Router Configuration

```rust
let router = LLMRouterBuilder::new()
    .provider(ProviderConfig::new(LLMProvider::OpenAI)
        .with_api_key_env("OPENAI_API_KEY")
        .with_model("gpt-4o-mini")
        .with_priority(0))  // Primary
    .provider(ProviderConfig::new(LLMProvider::Anthropic)
        .with_api_key_env("ANTHROPIC_API_KEY")
        .with_priority(1))  // Fallback
    .strategy(RoutingStrategy::Primary)
    .fallback_chain(vec![LLMProvider::OpenAI, LLMProvider::Anthropic])
    .build()?;
```

### Making Requests

```rust
let messages = vec![
    ChatMessage::system("You are helpful."),
    ChatMessage::user("Hello!"),
];

let response = router.chat(&messages, None).await?;
println!("Response: {}", response.content);
```

## Routing Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `Primary` | Use primary, fallback on failure | Production reliability |
| `LoadBalance` | Distribute across providers | High throughput |
| `CostOptimized` | Prefer cheaper providers | Budget constraints |
| `QualityOptimized` | Prefer best providers | Critical tasks |
| `LatencyOptimized` | Prefer fastest providers | Real-time apps |
| `Hybrid` | Balance all factors | General use |

## Circuit Breaker

Automatic failure detection and recovery:

```rust
// Configure circuit breaker
.circuit_breaker_threshold(5)   // Trip after 5 failures
.circuit_breaker_reset(Duration::from_secs(60))  // Reset after 60s

// Check state
let state = router.circuit_state(LLMProvider::OpenAI);
// States: Closed (normal), Open (tripped), HalfOpen (testing)
```

## Cost Tracking

Monitor spending across providers:

```rust
let snapshot = router.cost_tracker().snapshot();
println!("Total cost: ${:.4}", snapshot.estimated_cost);
println!("By provider:");
for (provider, stats) in &snapshot.by_provider {
    println!("  {:?}: ${:.4}", provider, stats.cost);
}
```

## Streaming

```rust
use futures::StreamExt;

let mut stream = router.chat_stream(&messages, None).await?;

while let Some(chunk) = stream.next().await {
    print!("{}", chunk?);
}
```

## Provider Models

### OpenAI
- `Gpt4o` - GPT-4o (best)
- `Gpt4oMini` - GPT-4o Mini (fast, cheap)
- `Gpt4Turbo` - GPT-4 Turbo

### Anthropic
- `Claude3Opus` - Most capable
- `Claude3Sonnet` - Balanced
- `Claude3Haiku` - Fastest

### Ollama (Local)
- `Llama3_2` - Llama 3.2
- `Mistral` - Mistral 7B
- `CodeLlama` - Code-specialized

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OLLAMA_HOST` | Ollama endpoint (default: localhost:11434) |

## Next Steps

- See [rest-api-client](../rest-api-client/) for HTTP server
- Explore [multi-agent](../multi-agent/) for complex workflows
- Check the `llm` module for advanced client options
