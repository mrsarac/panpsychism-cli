# Getting Started with Panpsychism

> Your first steps with the Sorcerer's Wand

This guide will help you install Panpsychism, run your first query, and understand the basic concepts.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Basic Configuration](#basic-configuration)
5. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **Rust**: 1.70 or later
- **OS**: macOS, Linux, or Windows
- **Memory**: 2GB RAM minimum (4GB recommended)
- **Disk**: 500MB for dependencies and data

### LLM Provider

You need access to an LLM provider. Choose one:

| Provider | Setup | Cost |
|----------|-------|------|
| **Antigravity Proxy** (Recommended) | Local proxy, Google account | Free |
| **Gemini API** | API key from Google AI Studio | Pay-per-use |
| **Ollama** | Local installation | Free |
| **OpenAI** | API key from OpenAI | Pay-per-use |
| **Anthropic** | API key from Anthropic | Pay-per-use |

---

## Installation

### Option 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/mrsarac/prompt-library.git
cd prompt-library/panpsychism

# Build release binary
cargo build --release

# The binary is at ./target/release/panpsychism
# Optionally, copy to your PATH:
cp target/release/panpsychism /usr/local/bin/
```

### Option 2: Cargo Install (Coming Soon)

```bash
cargo install panpsychism
```

### Verify Installation

```bash
panpsychism --version
# Output: panpsychism 1.0.0
```

---

## Quick Start

### Step 1: Set Up LLM Provider

**Option A: Antigravity Proxy (Free, Recommended)**

```bash
# macOS
brew tap lbjlaq/antigravity-manager
brew install --cask --no-quarantine antigravity-tools

# Start from menu bar and login with Google account
# Default endpoint: http://127.0.0.1:8045
```

**Option B: Direct API Key**

```bash
# Gemini
export GEMINI_API_KEY="your-api-key-here"

# Or OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Or Anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Step 2: Create Your First Prompt

```bash
# Create prompts directory
mkdir -p prompts

# Create a sample prompt
cat > prompts/hello-world.md << 'EOF'
---
id: hello-001
title: "Hello World Assistant"
category: general
tags:
  - greeting
  - introduction
privacy_tier: public
version: "1.0.0"
---

# Hello World Assistant

You are a friendly assistant who helps users get started with new tools and technologies.

## Guidelines

- Be welcoming and encouraging
- Provide clear, step-by-step instructions
- Use simple language
- Include practical examples

## Example Interaction

User: "How do I get started?"
Assistant: "Welcome! Let me guide you through the basics..."
EOF
```

### Step 3: Index Your Prompts

```bash
panpsychism index --dir ./prompts --output ./data/prompts.mv2

# Expected output:
# Scanning prompts directory: ./prompts
# Indexing complete!
#    Directory: ./prompts
#    Indexed: 1 prompts
#    Output: ./data/prompts.mv2
#    Duration: 0.05s
```

### Step 4: Search for Prompts

```bash
panpsychism search "greeting"

# Expected output:
# Searching for: "greeting"
#
#  1. Hello World Assistant                    [92.3%]
#     Tags: greeting, introduction
#
# Found 1 result in 0.02s
```

### Step 5: Ask a Question

```bash
panpsychism ask "How do I get started with Rust?"

# The system will:
# 1. Search for relevant prompts
# 2. Analyze your intent
# 3. Select an orchestration strategy
# 4. Build a meta-prompt
# 5. Call the LLM
# 6. Validate with Spinoza principles
# 7. Return the refined response
```

### Step 6: Start Interactive Shell

```bash
panpsychism shell

#   ____                              _     _
#  |  _ \ __ _ _ __  _ __  ___ _   _  ___| |__ (_)___ _ __ ___
#  | |_) / _` | '_ \| '_ \/ __| | | |/ __| '_ \| / __| '_ ` _ \
#  |  __/ (_| | | | | |_) \__ \ |_| | (__| | | | \__ \ | | | | |
#  |_|   \__,_|_| |_| .__/|___/\__, |\___|_| |_|_|___/_| |_| |_|
#                   |_|        |___/
#
#   Interactive Shell v1.0.0
#   Type /help for commands, or ask a question.
#
# > /help
# > How do I implement OAuth2?
# > /exit
```

---

## Basic Configuration

### Configuration File Location

| Platform | Path |
|----------|------|
| macOS | `~/.config/panpsychism/config.yaml` |
| Linux | `~/.config/panpsychism/config.yaml` |
| Windows | `%APPDATA%/panpsychism/config.yaml` |

### Minimal Configuration

```yaml
# ~/.config/panpsychism/config.yaml

# Prompt library location
prompts_dir: ./prompts
data_dir: ./data
index_file: ./data/prompts.mv2

# LLM Configuration
llm_endpoint: http://127.0.0.1:8045/v1/chat/completions
llm_api_key: sk-antigravity
llm_model: gemini-3-flash

# Privacy settings
privacy:
  tier: local
  share_patterns: false
  share_ratings: false
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `ANTIGRAVITY_API_KEY` | Antigravity proxy key | `sk-antigravity` |
| `RUST_LOG` | Log level | `info`, `debug`, `trace` |

---

## Next Steps

Now that you have Panpsychism running, explore these resources:

### Learn More

- **[User Guide](USER_GUIDE.md)** - Comprehensive guide to all features
- **[CLI Reference](CLI_REFERENCE.md)** - All commands and options
- **[Prompt Library Guide](PROMPT_LIBRARY.md)** - Create and manage prompts

### Build Your Prompt Library

1. Review the example prompts in `prompts/`
2. Create prompts for your specific use cases
3. Organize with categories and tags
4. Re-index when you add new prompts

### Integrate with Your Workflow

- **[API Reference](API_REFERENCE.md)** - Build applications with the REST API
- **[Architecture](ARCHITECTURE.md)** - Understand the system design

### Join the Community

- GitHub: [mrsarac/prompt-library](https://github.com/mrsarac/prompt-library)
- Issues: Report bugs and request features
- Discussions: Share prompts and best practices

---

## Troubleshooting

### "LLM API key not found"

```bash
# Check if Antigravity is running
curl http://127.0.0.1:8045/v1/models

# Or set API key directly
export GEMINI_API_KEY="your-key"
```

### "No prompts found"

```bash
# Verify prompts directory
ls -la prompts/

# Check prompt file format
head -20 prompts/your-prompt.md
# Must have YAML frontmatter with ---
```

### "Connection refused"

```bash
# If using Antigravity, ensure it's running
# Check menu bar for Antigravity icon

# If using direct API, check endpoint
curl https://generativelanguage.googleapis.com/v1/models
```

### Build Errors

```bash
# Update Rust
rustup update stable

# Clean and rebuild
cargo clean
cargo build --release
```

---

*Ready to dive deeper? Continue with the [User Guide](USER_GUIDE.md).*
