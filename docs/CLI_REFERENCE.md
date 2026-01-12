# Panpsychism CLI Reference

> Complete command-line interface documentation

This document provides detailed information about all CLI commands, options, and usage patterns.

---

## Table of Contents

1. [Installation](#installation)
2. [Global Options](#global-options)
3. [Commands](#commands)
4. [Shell Mode](#shell-mode)
5. [Environment Variables](#environment-variables)
6. [Exit Codes](#exit-codes)
7. [Examples](#examples)

---

## Installation

### From Source

```bash
git clone https://github.com/mrsarac/prompt-library.git
cd prompt-library/panpsychism
cargo build --release

# Add to PATH
cp target/release/panpsychism /usr/local/bin/
```

### Verify Installation

```bash
panpsychism --version
# panpsychism 1.0.0

panpsychism --help
```

---

## Global Options

Options that apply to all commands:

```
OPTIONS:
    --json              Output logs in JSON format
    --log-level <LEVEL> Set log level [default: info]
                        Values: error, warn, info, debug, trace
    -h, --help          Print help information
    -V, --version       Print version information
```

### Examples

```bash
# JSON log output (for parsing)
panpsychism --json search "auth"

# Debug logging
panpsychism --log-level debug ask "question"

# Trace logging (most verbose)
RUST_LOG=trace panpsychism ask "question"
```

---

## Commands

### panpsychism index

Index prompts from a directory into a searchable format.

```
USAGE:
    panpsychism index [OPTIONS]

OPTIONS:
    -d, --dir <DIR>       Prompts directory [default: ./prompts]
    -o, --output <FILE>   Output index file [default: ./data/masters.mv2]
    -f, --force           Force re-index even if unchanged
    -v, --verbose         Show detailed indexing progress
    -h, --help            Print help
```

**Examples:**

```bash
# Index default directory
panpsychism index

# Index custom directory
panpsychism index --dir /path/to/prompts

# Custom output file
panpsychism index --dir ./prompts --output ./data/library.mv2

# Force re-index with verbose output
panpsychism index --force --verbose
```

**Output:**

```
Scanning prompts directory: ./prompts
Indexing complete!
   Directory: ./prompts
   Indexed: 42 prompts
   Skipped: 2 files (invalid format)
   Output: ./data/masters.mv2
   Duration: 0.34s
   Categories: security, development, writing
   Tags: 156 unique
```

---

### panpsychism search

Search for prompts in the indexed library.

```
USAGE:
    panpsychism search [OPTIONS] <QUERY>

ARGS:
    <QUERY>    Search query (keywords)

OPTIONS:
    -t, --top <N>           Number of results [default: 5]
    -c, --category <CAT>    Filter by category
    --tags <TAGS>           Filter by tags (comma-separated)
    --min-score <SCORE>     Minimum relevance score [default: 0.1]
    -v, --verbose           Show detailed match information
    -o, --output <FORMAT>   Output format: table, json, yaml [default: table]
    -h, --help              Print help
```

**Examples:**

```bash
# Basic search
panpsychism search "authentication"

# More results
panpsychism search "security" --top 10

# Filter by category
panpsychism search "api" --category development

# Filter by tags
panpsychism search "code" --tags "review,security"

# JSON output
panpsychism search "kubernetes" --output json

# Verbose with low threshold
panpsychism search "database" --min-score 0.01 --verbose
```

**Output (Table):**

```
Searching for: "authentication"

 # | Title                           | Score  | Category   | Tags
---+---------------------------------+--------+------------+------------------
 1 | OAuth2 Authentication Guide     | 87.5%  | security   | oauth, jwt, auth
 2 | JWT Token Validation            | 72.3%  | security   | jwt, validation
 3 | Session Management              | 45.1%  | security   | session, auth

Found 3 results in 0.05s
```

**Output (JSON):**

```json
{
  "query": "authentication",
  "results": [
    {
      "id": "auth-001",
      "title": "OAuth2 Authentication Guide",
      "score": 0.875,
      "category": "security",
      "tags": ["oauth", "jwt", "auth"]
    }
  ],
  "total": 3,
  "latency_ms": 50
}
```

---

### panpsychism ask

Ask a question with orchestrated prompt selection.

```
USAGE:
    panpsychism ask [OPTIONS] <QUESTION>

ARGS:
    <QUESTION>    Your question or query

OPTIONS:
    -v, --verbose           Show reasoning trace
    -s, --strategy <STR>    Force strategy: focused, ensemble, chain, parallel
    --agent <AGENT>         Use specific agent
    --no-validate           Skip Spinoza validation
    --max-tokens <N>        Maximum response tokens [default: 2048]
    --temperature <T>       LLM temperature 0.0-1.0 [default: 0.7]
    -o, --output <FORMAT>   Output format: text, json, markdown [default: text]
    -h, --help              Print help
```

**Examples:**

```bash
# Basic question
panpsychism ask "How do I implement OAuth2?"

# With verbose reasoning trace
panpsychism ask "Best practices for API design" --verbose

# Force specific strategy
panpsychism ask "Compare REST vs GraphQL" --strategy ensemble

# Use specific agent
panpsychism ask "Create a code review prompt" --agent synthesizer

# JSON output
panpsychism ask "Explain microservices" --output json

# Lower temperature (more deterministic)
panpsychism ask "What is JWT?" --temperature 0.3
```

**Output (Normal):**

```
To implement OAuth2 authentication, follow these steps:

1. **Register your application** with the OAuth provider (Google, GitHub, etc.)
   to obtain client credentials.

2. **Implement the authorization flow**:
   - Redirect users to the provider's authorization endpoint
   - Handle the callback with the authorization code
   - Exchange the code for access and refresh tokens

3. **Secure token storage**:
   - Store refresh tokens securely (encrypted)
   - Keep access tokens in memory when possible
   - Implement token refresh logic

4. **Validate tokens** on protected routes...
```

**Output (Verbose):**

```
[SEARCH] Searching for relevant prompts...
  Found: auth-001 (0.87), security-002 (0.65)

[ANALYZE] Analyzing query intent...
  Category: security
  Complexity: 6/10
  Keywords: oauth2, implement, authentication

[STRATEGY] Selected: Focused
  Reason: Specific technical question, single best prompt sufficient

[SYNTHESIZE] Building meta-prompt...
  Base prompt: auth-001
  Context tokens: 256

[LLM] Calling Gemini API...
  Model: gemini-3-flash
  Tokens: 512

[VALIDATE] Spinoza validation...
  Conatus: 0.85 (growth-focused)
  Ratio: 0.92 (logical)
  Laetitia: 0.78 (positive)
  Status: PASSED

[RESPONSE]
To implement OAuth2 authentication, follow these steps...
```

---

### panpsychism shell

Start the interactive shell mode.

```
USAGE:
    panpsychism shell [OPTIONS]

OPTIONS:
    -v, --verbose           Start with verbose mode on
    --no-history            Don't save command history
    --history-file <PATH>   Custom history file path
    -h, --help              Print help
```

**Example:**

```bash
panpsychism shell
```

See [Shell Mode](#shell-mode) for detailed shell documentation.

---

### panpsychism serve

Start the REST API server.

```
USAGE:
    panpsychism serve [OPTIONS]

OPTIONS:
    -p, --port <PORT>       Server port [default: 8080]
    -h, --host <HOST>       Bind address [default: 127.0.0.1]
    --cors                  Enable CORS for all origins
    --tls-cert <PATH>       TLS certificate file
    --tls-key <PATH>        TLS key file
    -h, --help              Print help
```

**Examples:**

```bash
# Start on default port
panpsychism serve

# Custom port and host
panpsychism serve --port 3000 --host 0.0.0.0

# With TLS
panpsychism serve --tls-cert cert.pem --tls-key key.pem
```

**Output:**

```
Starting Panpsychism API Server...
  Version: 1.0.0
  Address: http://127.0.0.1:8080
  Agents: 40 loaded
  Prompts: 42 indexed

API Endpoints:
  POST /api/v1/query          - Process queries
  GET  /api/v1/agents         - List agents
  GET  /api/v1/health         - Health check
  WS   /api/v1/stream         - WebSocket streaming

Server ready.
```

---

### panpsychism status

Show system status and health.

```
USAGE:
    panpsychism status [OPTIONS]

OPTIONS:
    --json                  JSON output
    -v, --verbose           Detailed status
    -h, --help              Print help
```

**Example:**

```bash
panpsychism status
```

**Output:**

```
Panpsychism Status
==================

Version: 1.0.0
Uptime: 2h 34m

Components:
  Agent Bus      [OK]  40/40 agents active
  Memory Layer   [OK]  128MB used
  LLM Router     [OK]  3 providers configured
  Prompt Store   [OK]  42 prompts indexed

LLM Providers:
  Gemini         [OK]  gemini-3-flash
  OpenAI         [OK]  gpt-4o
  Ollama         [OK]  llama3.2

Statistics:
  Total Queries: 1,234
  Success Rate: 99.2%
  Avg Latency: 1.2s
```

---

### panpsychism agents

List and manage agents.

```
USAGE:
    panpsychism agents [SUBCOMMAND]

SUBCOMMANDS:
    list      List all agents
    info      Show agent details
    status    Show agent status
    help      Print help
```

**Examples:**

```bash
# List all agents
panpsychism agents list

# Agent details
panpsychism agents info synthesizer

# Agent status
panpsychism agents status validator
```

**Output (list):**

```
Panpsychism Agents (40 total)
=============================

Tier 8: Masters (Meta-Coordination)
  [36] Transcender      Supreme Orchestrator    [ACTIVE]
  [37] Evolver          System Evolution        [ACTIVE]
  [38] Harmonizer       Balance Keeper          [ACTIVE]
  [39] Federator        Distributed Coordinator [ACTIVE]
  [40] Consciousness    Meta-Awareness          [ACTIVE]

Tier 7: Architects (Structure & Design)
  [31] Composer         Response Composer       [ACTIVE]
  [32] Templater        Template Manager        [ACTIVE]
  ...

Tier 1: Core (Core Operations)
  [1] Orchestrator      Query Coordinator       [ACTIVE]
  [2] Search            Prompt Search           [ACTIVE]
  [3] Synthesizer       Meta-Prompt Builder     [ACTIVE]
  [4] Validator         Spinoza Validation      [ACTIVE]
  [5] Corrector         Quality Control         [ACTIVE]
```

---

### panpsychism config

Manage configuration.

```
USAGE:
    panpsychism config [SUBCOMMAND]

SUBCOMMANDS:
    show      Show current configuration
    set       Set a configuration value
    reset     Reset to defaults
    path      Show config file path
    help      Print help
```

**Examples:**

```bash
# Show configuration
panpsychism config show

# Set a value
panpsychism config set llm_model "gpt-4o"

# Show config path
panpsychism config path
```

---

### panpsychism batch

Process multiple queries in batch mode.

```
USAGE:
    panpsychism batch [OPTIONS]

OPTIONS:
    -i, --input <FILE>      Input file (JSON array of queries)
    -o, --output <FILE>     Output file for results
    --parallel <N>          Parallel processing [default: 4]
    --progress              Show progress bar
    -h, --help              Print help
```

**Example:**

```bash
# Create input file
cat > queries.json << 'EOF'
[
  {"query": "How to implement OAuth2?"},
  {"query": "Best practices for API design"},
  {"query": "Kubernetes networking explained"}
]
EOF

# Process batch
panpsychism batch --input queries.json --output results.json --progress
```

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
| `/history` | Show command history |
| `/clear` | Clear screen |
| `/exit` or `/quit` | Exit shell |

### Shell Usage

```
  ____                              _     _
 |  _ \ __ _ _ __  _ __  ___ _   _  ___| |__ (_)___ _ __ ___
 | |_) / _` | '_ \| '_ \/ __| | | |/ __| '_ \| / __| '_ ` _ \
 |  __/ (_| | | | | |_) \__ \ |_| | (__| | | | \__ \ | | | | |
 |_|   \__,_|_| |_| .__/|___/\__, |\___|_| |_|_|___/_| |_| |_|
                  |_|        |___/

  Interactive Shell v1.0.0
  Type /help for commands, or ask a question.

> How do I implement authentication?
[Processing...]

To implement authentication, you should consider...

> /verbose
Verbose mode: ON

> /stats
Session Statistics:
  Queries: 5
  Cache Hits: 2
  Avg Latency: 1.2s
  Tokens Used: 2,048

> /exit
Goodbye!
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit shell |
| `Up/Down` | Navigate history |
| `Tab` | Auto-complete commands |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PANPSYCHISM_CONFIG` | Config file path | `~/.config/panpsychism/config.yaml` |
| `PANPSYCHISM_PROMPTS_DIR` | Prompts directory | `./prompts` |
| `PANPSYCHISM_DATA_DIR` | Data directory | `./data` |
| `PANPSYCHISM_LOG_LEVEL` | Log level | `info` |
| `PANPSYCHISM_LOG_JSON` | JSON log format | `0` |
| `GEMINI_API_KEY` | Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `ANTIGRAVITY_API_KEY` | Antigravity proxy key | `sk-antigravity` |
| `RUST_LOG` | Rust logging directive | - |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |
| `3` | Configuration error |
| `4` | LLM provider error |
| `5` | Index not found |
| `6` | Network error |
| `130` | Interrupted (Ctrl+C) |

---

## Examples

### Daily Workflow

```bash
# Morning: Update index
panpsychism index --dir ./prompts

# Search for relevant prompts
panpsychism search "code review"

# Ask questions
panpsychism ask "How to review Rust code for memory safety?"

# Interactive session
panpsychism shell
```

### Scripting

```bash
#!/bin/bash
# Script to process queries from a list

while read -r query; do
    echo "Processing: $query"
    panpsychism ask "$query" --output json >> results.jsonl
done < queries.txt
```

### CI/CD Integration

```bash
# In CI pipeline
panpsychism ask "Review this code for security issues" \
    --output json \
    --no-validate \
    --max-tokens 1024
```

### Piping

```bash
# Pipe output to other tools
panpsychism search "kubernetes" --output json | jq '.results[0].title'

# Chain with other commands
cat code.rs | panpsychism ask "Review this Rust code" --output markdown
```

---

*For API integration, see [API Reference](API_REFERENCE.md). For prompt management, see [Prompt Library](PROMPT_LIBRARY.md).*
