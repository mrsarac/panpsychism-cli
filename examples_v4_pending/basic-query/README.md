# Basic Query Example

The simplest way to use Panpsychism for processing queries.

## What This Example Does

1. Loads configuration from environment
2. Initializes core components (search, validator, synthesizer)
3. Creates an orchestrator with default settings
4. Processes a user query
5. Displays the response with metadata

## Running

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the example
cargo run --example basic-query
```

## Expected Output

```
=== Panpsychism Basic Query Example ===

Query: How do I implement authentication in a Rust web application?

Processing...

=== Response ===

[Synthesized response about Rust authentication...]

=== Metadata ===
Strategy: Focused
Prompts used: 3
Confidence: 0.87
Processing time: 1.234s
```

## Key Concepts

### The Sorcerer's Wand Metaphor

- **Sorcerer (You)**: Speaks an incantation (query)
- **Grimoire (SearchEngine)**: Contains spells (prompts)
- **Wand (Orchestrator)**: Channels the magic
- **Creation**: The final response

### Configuration

The `Config::from_env()` reads:
- `OPENAI_API_KEY` - OpenAI API key
- `PROMPTS_DIR` - Directory containing prompt files
- `PANPSYCHISM_LOG_LEVEL` - Logging level (debug, info, warn, error)

### Strategies

- `Strategy::Focused` - Single best prompt (default)
- `Strategy::Ensemble` - Multiple parallel prompts
- `Strategy::Chain` - Sequential prompts
- `Strategy::Parallel` - Merged prompts

## Code Structure

```rust
// 1. Initialize components
let config = Config::from_env()?;
let orchestrator = Orchestrator::quick(&config).await?;

// 2. Process query
let response = orchestrator.process("your query").await?;

// 3. Use response
println!("{}", response.content);
```

## Next Steps

- Try different strategies with `query_with_strategy()`
- Explore [multi-agent](../multi-agent/) for complex workflows
- See [prompt-library](../prompt-library/) for custom templates
