# Getting Started with Panpsychism

Welcome to Panpsychism! This guide will help you get up and running quickly.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Rust** | 1.70+ | Install via [rustup](https://rustup.rs) |
| **Gemini API Key** | - | Get free at [Google AI Studio](https://aistudio.google.com/apikey) |

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/mrsarac/panpsychism-cli
cd panpsychism-cli
cargo install --path .
```

### From Crates.io (Coming Soon)

```bash
cargo install panpsychism
```

## Configuration

### 1. Set Your API Key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Add to your shell profile (`~/.zshrc` or `~/.bashrc`) for persistence.

### 2. Create Config File (Optional)

```bash
mkdir -p ~/.config/panpsychism
```

Create `~/.config/panpsychism/config.yaml`:

```yaml
prompts_dir: ./prompts
data_dir: ./data
index_file: ./data/masters.mv2

# Gemini API Configuration
llm_endpoint: https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
llm_api_key: ${GEMINI_API_KEY}
llm_model: gemini-2.0-flash

privacy:
  default_tier: internal
```

## Quick Test

### 1. Index Your Prompts

```bash
panpsychism index --dir ./prompts
```

### 2. Search for Prompts

```bash
panpsychism search "authentication"
```

### 3. Ask a Question

```bash
panpsychism ask "How do I implement OAuth2?" --verbose
```

## Next Steps

- Read the [User Guide](USER_GUIDE.md) for detailed usage
- Explore the [API Reference](API_REFERENCE.md) for library usage
- Check [Architecture](ARCHITECTURE.md) to understand the system
