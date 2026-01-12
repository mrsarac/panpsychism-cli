# Panpsychism Examples

This directory contains example applications demonstrating various features of the Panpsychism library.

## Prerequisites

```bash
# Set your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use local Ollama
ollama serve
ollama pull llama3.2
```

## Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| [basic-query](./basic-query/) | Simple query processing | Orchestrator, basic config |
| [multi-agent](./multi-agent/) | Agent orchestration | Bus, multiple agents |
| [prompt-library](./prompt-library/) | Prompt templates | PromptStore, templating |
| [llm-router](./llm-router/) | Provider routing | OpenAI, Anthropic, Ollama |
| [rest-api-client](./rest-api-client/) | REST API server | HTTP endpoints, WebSocket |

## Running Examples

Each example is a standalone Rust binary:

```bash
# Navigate to the panpsychism root
cd panpsychism

# Run a specific example
cargo run --example basic-query

# Run with release optimizations
cargo run --release --example multi-agent

# Run with logging
RUST_LOG=debug cargo run --example llm-router
```

## Project Structure

```
examples/
├── basic-query/
│   ├── main.rs          # Entry point
│   └── README.md        # Example documentation
├── multi-agent/
│   ├── main.rs
│   └── README.md
├── prompt-library/
│   ├── main.rs
│   └── README.md
├── llm-router/
│   ├── main.rs
│   └── README.md
└── rest-api-client/
    ├── main.rs
    └── README.md
```

## Adding Examples to Cargo.toml

To run these examples with `cargo run --example`, add to `Cargo.toml`:

```toml
[[example]]
name = "basic-query"
path = "examples/basic-query/main.rs"

[[example]]
name = "multi-agent"
path = "examples/multi-agent/main.rs"

[[example]]
name = "prompt-library"
path = "examples/prompt-library/main.rs"

[[example]]
name = "llm-router"
path = "examples/llm-router/main.rs"

[[example]]
name = "rest-api-client"
path = "examples/rest-api-client/main.rs"
```

## Philosophy

These examples follow Panpsychism's core philosophy:

- **Sorcerer's Wand**: Your words become creation
- **Spinoza's Framework**: Conatus (drive), Natura (alignment), Ratio (logic), Laetitia (joy)
- **Privacy-First**: All examples respect data classification tiers

## License

MIT - See [LICENSE](../LICENSE) for details.
