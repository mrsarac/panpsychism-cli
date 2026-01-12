# Panpsychism User Guide

> Comprehensive guide to mastering the Sorcerer's Wand

This guide covers all aspects of using Panpsychism, from configuration to advanced features.

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Using Agents](#using-agents)
4. [Prompt Library](#prompt-library)
5. [LLM Providers](#llm-providers)
6. [Privacy & Security](#privacy--security)
7. [Shell Mode](#shell-mode)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What is Panpsychism?

Panpsychism is an intelligent prompt orchestration system built in Rust. It combines:

- **Semantic Search**: Find relevant prompts using TF-IDF scoring
- **Intent Analysis**: Understand query complexity and category
- **Smart Selection**: Choose optimal prompts based on strategy
- **Philosophical Validation**: Ensure responses align with Spinoza's principles
- **Self-Correction**: Detect and fix ambiguities automatically

### The Sorcerer Metaphor

| Concept | System Component |
|---------|------------------|
| **Sorcerer** | You (the user) |
| **Incantation** | Your query/intent |
| **Grimoire** | Prompt library |
| **Wand** | Orchestrator |
| **Spell** | Selected prompts |
| **Creation** | Final response |

### Core Workflow

```
Query → Search → Analyze Intent → Select Strategy → Synthesize → Validate → Respond
```

---

## Configuration

### Configuration File

The configuration file uses YAML format:

```yaml
# ~/.config/panpsychism/config.yaml

#============================================================================
# PATHS
#============================================================================

# Directory containing your prompt files (.md)
prompts_dir: ./prompts

# Directory for data files (indexes, cache, etc.)
data_dir: ./data

# Path to the indexed prompt file
index_file: ./data/prompts.mv2

#============================================================================
# LLM CONFIGURATION
#============================================================================

# LLM provider endpoint (OpenAI-compatible)
llm_endpoint: http://127.0.0.1:8045/v1/chat/completions

# API key for authentication
llm_api_key: sk-antigravity

# Model to use
llm_model: gemini-3-flash

# Request timeout in seconds
llm_timeout: 30

# Maximum tokens in response
llm_max_tokens: 4096

# Temperature for generation (0.0 = deterministic, 1.0 = creative)
llm_temperature: 0.7

#============================================================================
# PRIVACY CONFIGURATION
#============================================================================

privacy:
  # Privacy tier: local, hybrid, federated
  tier: local

  # Share usage patterns with community
  share_patterns: false

  # Share quality ratings
  share_ratings: false

  # K-anonymity level (higher = more privacy)
  anonymization_level: 10

#============================================================================
# SEARCH CONFIGURATION
#============================================================================

search:
  # Default number of results
  top_k: 5

  # Minimum relevance score (0.0 - 1.0)
  min_score: 0.1

  # Enable fuzzy matching
  fuzzy: true

  # Cache search results (LRU)
  cache_size: 100

#============================================================================
# VALIDATION CONFIGURATION
#============================================================================

validation:
  # Enable Spinoza philosophical validation
  enabled: true

  # Minimum scores for each principle (0.0 - 1.0)
  min_conatus: 0.3    # Self-preservation, growth
  min_ratio: 0.3      # Logical reasoning
  min_laetitia: 0.3   # Joy, positive affect

#============================================================================
# LOGGING CONFIGURATION
#============================================================================

logging:
  # Log level: error, warn, info, debug, trace
  level: info

  # Output format: human, json
  format: human

  # Log file path (optional)
  file: null
```

### Environment Variables

Environment variables override configuration file values:

| Variable | Description | Default |
|----------|-------------|---------|
| `PANPSYCHISM_CONFIG` | Config file path | `~/.config/panpsychism/config.yaml` |
| `PANPSYCHISM_PROMPTS_DIR` | Prompts directory | `./prompts` |
| `PANPSYCHISM_DATA_DIR` | Data directory | `./data` |
| `PANPSYCHISM_LOG_LEVEL` | Log level | `info` |
| `PANPSYCHISM_LOG_JSON` | JSON log format | `0` |

### LLM Provider Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTIGRAVITY_API_KEY` | Antigravity proxy key |
| `OLLAMA_ENDPOINT` | Ollama server URL |

---

## Using Agents

### The 40-Agent Architecture

Panpsychism uses a hierarchical system of 40 specialized agents organized into 8 tiers:

```
TIER 8: Masters      [36-40] — Meta-Coordination
TIER 7: Architects   [31-35] — Structure & Design
TIER 6: Guardians    [26-30] — Protection & Monitoring
TIER 5: Enchanters   [21-25] — Enhancement
TIER 4: Oracles      [16-20] — Prediction & Learning
TIER 3: Alchemists   [11-15] — Synthesis
TIER 2: Scholars     [6-10]  — Analysis
TIER 1: Masters      [1-5]   — Core Operations
```

### Key Agents

| Agent | Role | Description |
|-------|------|-------------|
| **Transcender** | Supreme Orchestrator | Routes queries, coordinates all agents |
| **Synthesizer** | Meta-Prompt Builder | Combines prompts into coherent instructions |
| **Validator** | Spinoza Guardian | Ensures philosophical alignment |
| **Corrector** | Quality Control | Detects and fixes ambiguities |
| **Predictor** | Intent Analyzer | Understands query complexity |
| **Recommender** | Strategy Selector | Chooses orchestration approach |
| **Learner** | Continuous Improvement | Adapts from feedback |

### Orchestration Strategies

The system automatically selects the best strategy:

| Strategy | When Used | Description |
|----------|-----------|-------------|
| **Focused** | Simple queries, 1-2 prompts | Single powerful prompt |
| **Ensemble** | Complex topics, multiple views | Combine perspectives |
| **Chain** | Multi-step workflows | Sequential processing |
| **Parallel** | Independent sub-tasks | Concurrent execution |

### Direct Agent Invocation

You can invoke specific agents directly:

```bash
# Use the synthesizer agent
panpsychism --agent synthesizer "Create a prompt for code review"

# Use the validator agent
panpsychism --agent validator "Check this response for quality"

# Use the predictor agent
panpsychism --agent predictor "Analyze this query's complexity"
```

---

## Prompt Library

### Prompt File Format

Prompts are Markdown files with YAML frontmatter:

```markdown
---
id: unique-identifier
title: "Human-Readable Title"
description: "Optional longer description"
category: category-name
tags:
  - tag1
  - tag2
  - tag3
privacy_tier: public
version: "1.0.0"
author: author-name
---

# Prompt Title

The actual prompt content goes here...

## Sections

You can organize with headers.

## Examples

Include code examples:

```python
def example():
    return "hello world"
```

## Usage Notes

Any additional instructions...
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `auth-001`) |
| `title` | string | Human-readable title |
| `category` | string | Category name |
| `tags` | list | Searchable tags |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | string | - | Longer description |
| `privacy_tier` | enum | `public` | Privacy level |
| `version` | string | `1.0.0` | Semantic version |
| `author` | string | - | Author name |

### Privacy Tiers

| Tier | Description | Network Access |
|------|-------------|----------------|
| `public` | Open to all | Full |
| `internal` | Organization only | Limited |
| `confidential` | Restricted access | Minimal |
| `restricted` | Highly sensitive | None |

### Indexing Prompts

```bash
# Index default directory
panpsychism index

# Index custom directory
panpsychism index --dir /path/to/prompts

# Custom output file
panpsychism index --dir ./prompts --output ./data/library.mv2
```

### Searching Prompts

```bash
# Basic search
panpsychism search "authentication"

# More results
panpsychism search "security" --top 10

# With verbose output
panpsychism search "kubernetes" -v
```

---

## LLM Providers

### Supported Providers

Panpsychism supports multiple LLM providers through a unified interface:

#### 1. Antigravity Proxy (Free)

```yaml
llm_endpoint: http://127.0.0.1:8045/v1/chat/completions
llm_api_key: sk-antigravity
llm_model: gemini-3-flash
```

Setup:
```bash
brew tap lbjlaq/antigravity-manager
brew install --cask --no-quarantine antigravity-tools
# Start from menu bar, login with Google
```

#### 2. Google Gemini

```yaml
llm_endpoint: https://generativelanguage.googleapis.com/v1beta/openai/
llm_api_key: ${GEMINI_API_KEY}
llm_model: gemini-2.0-flash
```

#### 3. OpenAI

```yaml
llm_endpoint: https://api.openai.com/v1/chat/completions
llm_api_key: ${OPENAI_API_KEY}
llm_model: gpt-4o
```

#### 4. Anthropic

```yaml
llm_endpoint: https://api.anthropic.com/v1/messages
llm_api_key: ${ANTHROPIC_API_KEY}
llm_model: claude-3-5-sonnet
```

#### 5. Ollama (Local)

```yaml
llm_endpoint: http://localhost:11434/v1/chat/completions
llm_api_key: ollama
llm_model: llama3.2
```

### LLM Router

The system includes an intelligent router for multiple providers:

```yaml
llm_routing:
  strategy: cost_optimized  # primary, load_balance, cost_optimized, quality
  fallback_chain:
    - gemini-3-flash      # Try this first
    - gpt-4o-mini         # Fallback 1
    - llama3.2            # Fallback 2 (local)
```

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| `primary` | Use first available provider |
| `load_balance` | Round-robin distribution |
| `cost_optimized` | Cheapest provider first |
| `quality_optimized` | Best model first |
| `hybrid` | Balance cost and quality |

---

## Privacy & Security

### Privacy Tiers

Panpsychism implements three privacy tiers:

| Tier | Data Handling | Network |
|------|---------------|---------|
| **Local** | All data stays on device | None |
| **Hybrid** | User controls what to share | Selective |
| **Federated** | Full collaboration with anonymization | Full |

### Local Tier (Default)

```yaml
privacy:
  tier: local
  share_patterns: false
  share_ratings: false
```

- All prompts and queries processed locally
- No network requests except to LLM provider
- Complete data isolation

### Hybrid Tier

```yaml
privacy:
  tier: hybrid
  share_patterns: true    # Share anonymous usage patterns
  share_ratings: true     # Share quality ratings
  anonymization_level: 10 # K-anonymity level
```

- Selective sharing with consent
- Anonymized usage statistics
- Quality ratings help improve system

### Federated Tier

```yaml
privacy:
  tier: federated
  share_patterns: true
  share_ratings: true
  anonymization_level: 20
```

- Full collaboration with community
- Differential privacy protection
- Collective knowledge building

### Security Features

- **No data persistence by default**: Queries not logged
- **API key encryption**: Keys stored securely
- **Rate limiting**: Prevent abuse
- **Input sanitization**: Protect against injection

---

## Shell Mode

### Starting the Shell

```bash
panpsychism shell
```

### Shell Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/config` | Display current configuration |
| `/verbose` | Toggle verbose mode |
| `/stats` | Show session statistics |
| `/agents` | List available agents |
| `/clear` | Clear screen |
| `/exit` | Exit shell |

### Interactive Features

```bash
> How do I implement OAuth2?
# System processes query and returns response

> /verbose
# Verbose mode: ON
# Now shows reasoning trace

> How do I implement OAuth2?
# Shows: Search → Analyze → Select → Synthesize → Validate → Respond

> /stats
# Session Statistics:
#   Queries: 5
#   Avg Latency: 1.2s
#   Cache Hits: 3
```

### History

- Use arrow keys to navigate history
- History persists across sessions
- Stored in `~/.panpsychism_history`

---

## Performance Tuning

### Caching

```yaml
search:
  cache_size: 100  # LRU cache for search results
```

### Batch Processing

```bash
# Process multiple queries
panpsychism batch --input queries.json --output results.json

# queries.json format:
# [
#   {"query": "How to implement auth?"},
#   {"query": "Best practices for API design"}
# ]
```

### Parallelization

```yaml
orchestration:
  max_parallel_agents: 5  # Concurrent agent execution
  timeout_ms: 30000       # Per-agent timeout
```

### Memory Management

```yaml
memory:
  short_term_size: 1000   # Items in short-term cache
  long_term_enabled: true # Enable persistent storage
```

---

## Troubleshooting

### Common Issues

#### "LLM API key not found"

```bash
# Check environment
echo $GEMINI_API_KEY

# Or check config
cat ~/.config/panpsychism/config.yaml | grep llm_api_key
```

#### "No prompts found"

```bash
# Verify prompts exist
ls -la prompts/

# Check index file
ls -la data/*.mv2

# Re-index
panpsychism index --dir ./prompts
```

#### "Search returns no results"

```bash
# Lower the minimum score
panpsychism search "query" --min-score 0.01

# Check prompt tags
grep -r "tags:" prompts/
```

#### "Slow responses"

```bash
# Enable debug logging
RUST_LOG=debug panpsychism ask "query"

# Check which step is slow
panpsychism ask "query" -v
```

#### "Validation fails"

```yaml
# Lower validation thresholds temporarily
validation:
  min_conatus: 0.1
  min_ratio: 0.1
  min_laetitia: 0.1
```

### Debug Mode

```bash
# Full trace logging
RUST_LOG=trace panpsychism ask "query"

# JSON logs for parsing
panpsychism --json ask "query"
```

### Getting Help

- GitHub Issues: [mrsarac/prompt-library/issues](https://github.com/mrsarac/prompt-library/issues)
- Documentation: [docs/](.)

---

*For API integration, see [API Reference](API_REFERENCE.md). For CLI details, see [CLI Reference](CLI_REFERENCE.md).*
