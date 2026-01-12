# Multi-Agent Orchestration Example

Demonstrates orchestrating multiple specialized agents through the AgentBus.

## What This Example Does

1. Creates an AgentBus for inter-agent communication
2. Initializes multiple specialized agents (enhancer, formatter, summarizer)
3. Registers agents with the bus
4. Processes input through a multi-step workflow
5. Shows bus statistics and message flow

## Architecture

```
+------------------+
|    AgentBus      |  <-- Central communication hub
+------------------+
         |
   +-----+-----+-----+
   |     |     |     |
   v     v     v     v
+------+------+------+------+
|Enhanc|Format|Summar|...   |
+------+------+------+------+
```

## Running

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the example
cargo run --example multi-agent

# With debug logging
RUST_LOG=debug cargo run --example multi-agent
```

## Expected Output

```
=== Panpsychism Multi-Agent Example ===

Input:
The authentication system needs to handle multiple OAuth providers...

Processing through multi-agent workflow...

[Coordinator] Starting workflow...
[Step 1] Enhancing input...
[Step 2] Formatting content...
[Step 3] Generating summary...
[Coordinator] Workflow completed with 3 messages

=== Final Result ===

[Summarized authentication requirements...]

=== Bus Statistics ===
Total messages: 3
Active agents: 3
Topics: ["workflow"]
```

## Key Concepts

### AgentBus

The central communication hub for all agents:

```rust
let bus = AgentBus::new(BusConfig::default());
bus.register(AgentId::new("agent1"), "Agent1").await?;
bus.publish(Topic::new("topic"), Message::text("key", "value")).await?;
```

### Specialized Agents

Each agent has a specific role:

- **EnhancerAgent**: Improves prompt quality
- **FormatterAgent**: Formats output (Markdown, JSON, etc.)
- **SummarizerAgent**: Creates concise summaries
- **ValidatorAgent**: Validates content against rules

### Workflow Patterns

1. **Sequential**: A -> B -> C
2. **Parallel**: A, B, C run simultaneously
3. **Fan-out/Fan-in**: One splits into many, then merges

## Code Patterns

### Sequential Workflow

```rust
let enhanced = enhancer.enhance(input).await?;
let formatted = formatter.format(&enhanced.content).await?;
let summary = summarizer.summarize(&formatted.content).await?;
```

### Parallel Execution

```rust
use futures::future::join_all;

let futures = inputs.iter().map(|i| agent.process(i));
let results = join_all(futures).await;
```

### Error-Resilient Pipeline

```rust
let result = match agent.process(input).await {
    Ok(r) => r,
    Err(e) => {
        eprintln!("Agent failed: {}", e);
        fallback_value
    }
};
```

## Available Agents

| Agent | Purpose |
|-------|---------|
| `EnhancerAgent` | Improve prompt quality |
| `FormatterAgent` | Format output |
| `SummarizerAgent` | Generate summaries |
| `ValidatorAgent` | Validate content |
| `PersonalizerAgent` | Personalize responses |
| `LocalizerAgent` | Localize content |
| `SanitizerAgent` | Remove sensitive data |

## Next Steps

- See [llm-router](../llm-router/) for provider routing
- Explore [rest-api-client](../rest-api-client/) for HTTP integration
- Check the `bus` module for advanced messaging patterns
