# Panpsychism v1.5.0 - Quick Usage Guide

> 🪄 The Sorcerer's Wand — Transform your words into creation.

## Prerequisites

### 1. Antigravity Proxy (Recommended - Free)
```bash
# macOS
brew tap lbjlaq/antigravity-manager
brew install --cask --no-quarantine antigravity-tools

# Start from menu bar, login with Google account
# Default: http://127.0.0.1:8045
```

### 2. Or Direct Gemini API Key
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Installation

```bash
cd /path/to/panpsychism

# Build
cargo build --release

# Or run directly
cargo run -- --help
```

## Quick Start

### Step 1: Create Prompts Directory
```bash
mkdir -p prompts

# Example prompt file (prompts/code-review.md)
cat > prompts/code-review.md << 'EOF'
---
title: Code Review Assistant
tags: [code, review, security]
privacy: public
---

You are a senior code reviewer. Analyze the provided code for:
- Security vulnerabilities
- Performance issues
- Best practices violations
- Potential bugs

Provide specific, actionable feedback.
EOF
```

### Step 2: Index Your Prompts
```bash
# Index prompts from ./prompts directory
cargo run -- index

# Custom directory
cargo run -- index -d /path/to/prompts -o ./data/my-library.mv2
```

### Step 3: Search Prompts
```bash
# Basic search
cargo run -- search "code review"

# Get more results
cargo run -- search "security" -t 10
```

### Step 4: Ask Questions (LLM Integration)
```bash
# Simple question
cargo run -- ask "How do I implement OAuth2?"

# With verbose reasoning trace
cargo run -- ask "Best practices for database migrations" -v
```

### Step 5: Interactive Shell
```bash
cargo run -- shell

# Shell commands:
#   help     - Show available commands
#   search   - Search prompts
#   ask      - Ask with LLM
#   clear    - Clear screen
#   exit     - Exit shell
```

## Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `index` | Index prompts into .mv2 file | `panpsychism index -d ./prompts` |
| `search` | Search for prompts | `panpsychism search "auth" -t 5` |
| `ask` | Ask question with orchestration | `panpsychism ask "How to...?" -v` |
| `shell` | Interactive mode | `panpsychism shell` |

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--json` | JSON log output | false |
| `--log-level` | error/warn/info/debug/trace | info |
| `-d, --dir` | Prompts directory (index) | ./prompts |
| `-o, --output` | Output .mv2 file (index) | ./data/masters.mv2 |
| `-t, --top` | Result count (search) | 5 |
| `-v, --verbose` | Show reasoning (ask) | false |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini API key | - |
| `ANTIGRAVITY_API_KEY` | Antigravity proxy key | sk-antigravity |
| `PANPSYCHISM_LOG_JSON` | Force JSON logging | 0 |

## Prompt File Format

```markdown
---
title: My Prompt Title
tags: [tag1, tag2, tag3]
privacy: public|internal|confidential|restricted
---

Your prompt content here...
```

## Example Session

```bash
# Terminal 1: Start Antigravity (if using)
# (Click menu bar icon, ensure logged in)

# Terminal 2: Run Panpsychism
cd ~/Documents/GitHub/prompt-library/panpsychism

# Index sample prompts
cargo run -- index
# Output: ✅ Indexed 5 prompts → ./data/masters.mv2

# Search
cargo run -- search "kubernetes"
# Output:
# 1. kubernetes-network-policy.md (0.87)
# 2. database-migration.md (0.23)

# Ask with LLM
cargo run -- ask "How do I set up a Kubernetes network policy?"
# Output: (LLM response with relevant prompts as context)

# Interactive shell
cargo run -- shell
🪄 Panpsychism Shell v1.5.0
> search oauth
> ask "Explain OAuth2 flow"
> exit
```

## Running Tests

```bash
# All tests
cargo test

# Specific module
cargo test search::
cargo test validator::

# With output
cargo test -- --nocapture
```

## Troubleshooting

### "API key not found"
```bash
# Option 1: Use Antigravity (free)
# Start Antigravity from menu bar

# Option 2: Set API key
export GEMINI_API_KEY="your-key"
```

### "No prompts found"
```bash
# Check prompts directory exists
ls -la prompts/

# Ensure .md files have YAML frontmatter
head -10 prompts/example.md
```

### "Connection refused"
```bash
# If using Antigravity, ensure it's running
curl http://127.0.0.1:8045/v1/models

# Check menu bar for Antigravity icon
```

## Architecture

```
🧙 You (Sorcerer)
    │
    │ "Incantation" (Query)
    ▼
┌──────────────┐
│  🪄 Wand     │ ← Orchestrator
└──────────────┘
    │
    ├── Search ──► Grimoire (Prompts)
    ├── Synthesize ──► Meta-prompt
    ├── Validate ──► Spinoza (CONATUS/RATIO/LAETITIA)
    └── Correct ──► Refinement
    │
    ▼
🎇 Creation (Response)
```

## More Information

- **Repository**: https://github.com/mrsarac/panpsychism
- **Version**: 1.5.0 (15 agents)
- **License**: Internal Use Only
