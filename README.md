# Panpsychism

> 🪄 The Sorcerer's Wand — Transform your words into creation

[![Crates.io](https://img.shields.io/crates/v/panpsychism)](https://crates.io/crates/panpsychism)
[![Documentation](https://docs.rs/panpsychism/badge.svg)](https://docs.rs/panpsychism)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

## Overview

**Panpsychism** is a Rust library and CLI tool for intelligent prompt orchestration. Like a sorcerer who shapes reality with words and a wand, Panpsychism transforms your voice commands into concrete creations. Your intent becomes an incantation, your prompt library is your grimoire, and the wand (this tool) channels them into magical results.

The system indexes your prompt library into a semantic memory, analyzes user intent, selects optimal prompts using various strategies, synthesizes responses through LLM integration, and validates outputs against Spinoza's philosophical principles.

Built on the foundation of Baruch Spinoza's Ethics, Panpsychism ensures that AI responses not only answer questions accurately but also align with principles of growth (CONATUS), reason (RATIO), and joy (LAETITIA).

## Features

- **Semantic Search** - Find relevant prompts using TF-IDF scoring with keyword matching
- **Intent Analysis** - Understand query complexity, category, and extract keywords
- **Smart Selection** - Choose optimal prompts based on strategy (Focused, Ensemble, Chain, Parallel)
- **Spinoza Validation** - Ensure responses align with philosophical principles
- **Self-Correction** - Detect ambiguities and improve responses through reflection
- **Privacy Tiers** - Local, Hybrid, or Federated data handling (GDPR compliant)
- **Interactive Shell** - REPL with history, stats, and verbose mode
- **Structured Logging** - JSON or human-readable logs with tracing

## Installation

### From Source

```bash
git clone https://github.com/your-username/panpsychism
cd panpsychism
cargo install --path .
```

### From Crates.io (coming soon)

```bash
cargo install panpsychism
```

### Requirements

- Rust 1.70 or later
- A Gemini API key (or Antigravity proxy for local development)

## Quick Start

### 1. Create Your Prompts Directory

```bash
mkdir -p prompts
```

Create prompt files with YAML frontmatter:

```markdown
---
id: auth-001
title: "Authentication Flow Guide"
category: security
tags:
  - authentication
  - oauth
  - jwt
privacy_tier: public
version: "1.0.0"
---

# Authentication Flow Guide

This prompt helps implement secure authentication flows...
```

### 2. Index Your Prompts

```bash
panpsychism index --dir ./prompts --output ./data/masters.mv2
```

Output:
```
Scanning prompts directory: ./prompts
Indexing complete!
   Directory: ./prompts
   Indexed: 5 prompts
   Skipped: 0 files
   Output: ./data/masters.mv2
   Duration: 0.12s
   5 unique tags, 3 categories
```

### 3. Search for Prompts

```bash
panpsychism search "authentication"
```

Output:
```
Searching for: "authentication"

 1. Authentication Flow Guide                    [87.5%]
    Tags: authentication, oauth, jwt

Found 1 result in 0.05s
```

### 4. Ask a Question

```bash
export GEMINI_API_KEY=your-key
panpsychism ask "How do I implement OAuth2?"
```

The pipeline:
1. Searches for relevant prompts
2. Analyzes your intent
3. Determines orchestration strategy
4. Builds a meta-prompt with Spinoza principles
5. Calls the LLM
6. Validates the response
7. Applies corrections if needed

### 5. Interactive Shell

```bash
panpsychism shell
```

```
  ____                              _     _
 |  _ \ __ _ _ __  _ __  ___ _   _  ___| |__ (_)___ _ __ ___
 | |_) / _` | '_ \| '_ \/ __| | | |/ __| '_ \| / __| '_ ` _ \
 |  __/ (_| | | | | |_) \__ \ |_| | (__| | | | \__ \ | | | | |
 |_|   \__,_|_| |_| .__/|___/\__, |\___|_| |_|_|___/_| |_| |_|
                  |_|        |___/

  Interactive Shell v0.1.0
  Type /help for commands, or ask a question.

> How do I handle JWT tokens securely?
```

**Shell Commands:**
- `/help` - Show available commands
- `/config` - Display current configuration
- `/verbose` - Toggle verbose mode (shows pipeline steps)
- `/stats` - Show session statistics
- `/clear` - Clear screen
- `/exit` - Exit shell

## Configuration

Configuration file location:
- **macOS/Linux**: `~/.config/panpsychism/config.yaml`
- **Windows**: `%APPDATA%/panpsychism/config.yaml`

### Example Configuration

```yaml
prompts_dir: ./prompts
data_dir: ./data
index_file: ./data/masters.mv2
llm_endpoint: http://127.0.0.1:8045/v1/chat/completions
llm_api_key: sk-antigravity
llm_model: gemini-3-flash
privacy:
  tier: local
  share_patterns: false
  share_ratings: false
  anonymization_level: 10
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | API key for Gemini (primary) |
| `ANTIGRAVITY_API_KEY` | API key for Antigravity proxy (fallback) |
| `RUST_LOG` | Log level (error, warn, info, debug, trace) |

## Prompt Format

Prompts are Markdown files with YAML frontmatter:

```markdown
---
id: unique-identifier
title: "Human-readable Title"
description: "Optional longer description"
category: category-name
tags:
  - tag1
  - tag2
privacy_tier: public  # public, internal, confidential, restricted
version: "1.0.0"
author: your-name
---

# Prompt Content

The actual prompt content goes here...

## Sections

You can use any Markdown formatting.

## Examples

Include code examples:

\`\`\`typescript
// Example code here
\`\`\`
```

### Privacy Tiers

| Tier | Description | Network |
|------|-------------|---------|
| `local` | All data stays on device | No |
| `hybrid` | User controls what to share | Selective |
| `federated` | Full collaboration with anonymization | Yes |

## Architecture

```text
🧙 Sorcerer (You)
       |
       | "Incantation" (Your intent/query)
       v
+------------------+
|   🪄 The Wand    |  <-- Orchestrator: channels your magic
+------------------+
       |
  +----+----+
  |         |
  v         v
+------+ +--------+
|Search| |Indexer |  <-- 📜 Grimoire: find the right spells (prompts)
+------+ +--------+
       |
       v
+------------------+
|   Synthesizer    |  <-- ⚗️ Combine spells into powerful magic
+------------------+
       |
       v
+------------------+
|    Validator     |  <-- 🏛️ Spinoza's blessing (philosophical validation)
+------------------+
       |
       v
+------------------+
|    Corrector     |  <-- ✨ Refine the spell if needed
+------------------+
       |
       v
  🎇 Creation (Final Response)
```

### Orchestration Strategies (Spell Types)

| Strategy | When Used | Description |
|----------|-----------|-------------|
| **Focused** | Simple incantations, 1-2 spells | Single powerful spell |
| **Ensemble** | Complex magic, multiple perspectives | Combine multiple spell schools |
| **Chain** | Multi-step rituals required | Sequential spell casting |
| **Parallel** | Independent enchantments | Simultaneous spellwork |

## Philosophy

Panpsychism is grounded in Baruch Spinoza's philosophical framework from his *Ethics* (1677):

### CONATUS (Self-Preservation)

> "Each thing, as far as it can by its own power, strives to persevere in its being."
> - Ethics III, Proposition 6

In validation, CONATUS measures whether content supports growth, learning, and positive development rather than destruction or harm. Content with high CONATUS scores contains words like *grow*, *learn*, *create*, *preserve*, *nurture*.

### RATIO (Reason)

> "It is in the nature of reason to perceive things under a certain form of eternity."
> - Ethics II, Proposition 44

RATIO validates logical consistency, coherent structure, and sound reasoning. High RATIO scores come from content with logical connectors (*therefore*, *because*, *consequently*), structured arguments, and evidence-based claims.

### LAETITIA (Joy)

> "Laetitia is a person's passage from a lesser to a greater perfection."
> - Ethics III, Proposition 11

LAETITIA measures whether content enhances positive affect, inspires growth, and moves toward greater understanding. Joy-enhancing terms like *hope*, *inspire*, *achieve*, *beautiful* contribute to high LAETITIA scores.

### Validation Example

```rust
use panpsychism::validator::SpinozaValidator;

let validator = SpinozaValidator::new();
let result = validator.validate("Learning brings joy and growth").await?;

// result.scores.conatus  ~= 0.7  (growth-focused)
// result.scores.ratio    ~= 0.5  (neutral)
// result.scores.laetitia ~= 0.8  (joy-enhancing)
// result.is_valid        == true
```

## Library Usage

```rust
use panpsychism::{Config, PrivacyTier};
use panpsychism::orchestrator::Orchestrator;
use panpsychism::search::{SearchEngine, SearchQuery, PromptMetadata};

#[tokio::main]
async fn main() -> panpsychism::Result<()> {
    // Load configuration
    let config = Config::load()?;

    // Create search engine with prompts
    let prompts = vec![
        PromptMetadata::new("auth-001", "Authentication Guide", "OAuth2 and JWT...", "prompts/auth.md")
            .with_tags(vec!["oauth".into(), "jwt".into()])
            .with_category("security"),
    ];
    let engine = SearchEngine::new(prompts);

    // Search for relevant prompts
    let query = SearchQuery::new("How to implement OAuth2?").with_top_k(5);
    let results = engine.search(&query).await?;

    // Use orchestrator for full pipeline
    let orchestrator = Orchestrator::new();
    let intent = orchestrator.analyze_intent("How to implement OAuth2?").await?;

    println!("Category: {}", intent.category);
    println!("Complexity: {}/10", intent.complexity);
    println!("Keywords: {:?}", intent.keywords);

    Ok(())
}
```

## CLI Reference

```bash
panpsychism [OPTIONS] <COMMAND>

Commands:
  index   Index prompts into .mv2 file
  search  Search for prompts
  ask     Ask a question with orchestrated prompts
  shell   Interactive shell mode

Options:
  --json           Output logs in JSON format
  --log-level      Set log level (error, warn, info, debug, trace) [default: info]
  -h, --help       Print help
  -V, --version    Print version
```

### Index Command

```bash
panpsychism index [OPTIONS]

Options:
  -d, --dir <DIR>       Prompts directory [default: ./prompts]
  -o, --output <FILE>   Output .mv2 file [default: ./data/masters.mv2]
```

### Search Command

```bash
panpsychism search [OPTIONS] <QUERY>

Options:
  -t, --top <N>   Number of results [default: 5]
```

### Ask Command

```bash
panpsychism ask [OPTIONS] <QUESTION>

Options:
  -v, --verbose   Show reasoning trace
```

## Development

### Running Tests

```bash
cargo test
```

### Building Documentation

```bash
cargo doc --open
```

### Code Structure

```
src/
  lib.rs          # Library entry point, re-exports
  main.rs         # CLI entry point
  cache.rs        # LRU cache for search results
  config.rs       # Configuration management
  constants.rs    # Global constants
  corrector.rs    # Ambiguity detection and correction
  error.rs        # Error types
  gemini.rs       # Gemini API client
  indexer.rs      # Prompt indexing
  orchestrator.rs # Strategy selection and prompt orchestration
  privacy.rs      # Privacy tier configuration
  recovery.rs     # Circuit breaker, retry, fallback
  search.rs       # TF-IDF search engine
  synthesizer.rs  # Response synthesis
  tracing_setup.rs# Logging configuration
  validator.rs    # Spinoza philosophical validation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Guidelines

- Write tests for new functionality
- Follow Rust conventions (run `cargo clippy`)
- Update documentation for API changes
- Keep the Spinoza philosophy in mind when adding features

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*"The mind's highest good is the knowledge of God, and the mind's highest virtue is to know God."*
- Baruch Spinoza, Ethics V, Proposition 28

Built with Rust and philosophical rigor.
