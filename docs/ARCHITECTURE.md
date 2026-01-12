# Panpsychism System Architecture

> Deep dive into the 40-agent system design

This document provides a comprehensive overview of Panpsychism's architecture, including the 8-tier agent hierarchy, communication bus, and infrastructure layers.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Agent Tiers](#agent-tiers)
3. [Agent Communication Bus](#agent-communication-bus)
4. [Memory Layer](#memory-layer)
5. [LLM Router](#llm-router)
6. [Request Flow](#request-flow)
7. [Spinoza Validation](#spinoza-validation)
8. [Infrastructure](#infrastructure)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │   CLI   │  │   API   │  │   SDK   │  │   Web   │  │WebSocket│       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
└───────┼────────────┼────────────┼────────────┼────────────┼─────────────┘
        │            │            │            │            │
        └────────────┴──────┬─────┴────────────┴────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│                         GATEWAY LAYER                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Transcender (Agent #40)                        │  │
│  │               Supreme Orchestrator & Request Router                │  │
│  │         Routes queries → Coordinates agents → Returns results      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│                      AGENT BUS LAYER (ACB)                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              Agent Communication Bus                               │  │
│  │      pub/sub • request/response • events • heartbeats             │  │
│  │                                                                    │  │
│  │  Topics: agent.*, query.*, system.*, metrics.*                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│                         AGENT TIERS                                      │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ TIER 8: Masters (36-40)         Meta-Coordination                │   │
│  │   Transcender, Evolver, Harmonizer, Federator, Consciousness     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ TIER 7: Architects (31-35)      Structure & Design               │   │
│  │   Composer, Templater, Documenter, Refactorer, Tester            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ TIER 6: Guardians (26-30)       Protection & Monitoring          │   │
│  │   Sanitizer, RateLimiter, Auditor, Monitor, Recoverer            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ TIER 5: Enchanters (21-25)      Enhancement                      │   │
│  │   Adapter, Localizer, Personalizer, Enhancer, Enricher           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ TIER 4: Oracles (16-20)         Prediction & Learning            │   │
│  │   Predictor, Recommender, Evaluator, Debugger, Learner           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ TIER 3: Alchemists (11-15)      Synthesis                        │   │
│  │   Synthesizer, Contextualizer, Formatter, Summarizer, Expander   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ TIER 2: Scholars (6-10)         Analysis                         │   │
│  │   ContentAnalyzer, Validator, Corrector, PromptSelector, Store   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ TIER 1: Core (1-5)              Core Operations                  │   │
│  │   Orchestrator, Search, Indexer, Cache, OutputRouter             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                                │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            │
│  │  Memory   │  │    LLM    │  │   Vector  │  │  Metrics  │            │
│  │   Layer   │  │   Router  │  │   Store   │  │ Collector │            │
│  │ (RocksDB) │  │ (Multi)   │  │  (TF-IDF) │  │(Prometheus│            │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Tiers

### Tier 1: Core Operations (Agents 1-5)

The foundational agents that handle basic operations.

| # | Agent | Role | Responsibility |
|---|-------|------|----------------|
| 1 | **Orchestrator** | Query Coordinator | Route queries, manage pipeline |
| 2 | **Search** | Prompt Finder | TF-IDF search, relevance scoring |
| 3 | **Indexer** | Library Manager | Index prompts, build vectors |
| 4 | **Cache** | Performance | LRU cache, result memoization |
| 5 | **OutputRouter** | Response Delivery | Format and deliver responses |

### Tier 2: Scholars (Agents 6-10)

Analysis and validation specialists.

| # | Agent | Role | Responsibility |
|---|-------|------|----------------|
| 6 | **ContentAnalyzer** | Content Analysis | Analyze query content, extract features |
| 7 | **Validator** | Spinoza Guardian | Validate against philosophical principles |
| 8 | **Corrector** | Quality Control | Detect ambiguities, suggest corrections |
| 9 | **PromptSelector** | Strategy Expert | Select optimal prompts and strategy |
| 10 | **PromptStore** | Library Storage | Manage prompt persistence |

### Tier 3: Alchemists (Agents 11-15)

Synthesis and transformation agents.

| # | Agent | Role | Responsibility |
|---|-------|------|----------------|
| 11 | **Synthesizer** | Meta-Prompt Builder | Combine prompts into instructions |
| 12 | **Contextualizer** | Context Manager | Build and manage context |
| 13 | **Formatter** | Output Formatter | Format responses (markdown, JSON) |
| 14 | **Summarizer** | Condensation | Summarize long content |
| 15 | **Expander** | Elaboration | Expand brief responses |

### Tier 4: Oracles (Agents 16-20)

Prediction, learning, and insight agents.

| # | Agent | Role | Responsibility |
|---|-------|------|----------------|
| 16 | **Predictor** | Intent Analyzer | Predict query complexity, category |
| 17 | **Recommender** | Strategy Advisor | Recommend orchestration approach |
| 18 | **Evaluator** | Quality Assessor | Evaluate response quality |
| 19 | **Debugger** | Error Analyzer | Diagnose and fix issues |
| 20 | **Learner** | Continuous Improvement | Learn from feedback |

### Tier 5: Enchanters (Agents 21-25)

Enhancement and adaptation specialists.

| # | Agent | Role | Responsibility |
|---|-------|------|----------------|
| 21 | **Adapter** | Format Adapter | Adapt between formats/protocols |
| 22 | **Localizer** | i18n Manager | Handle localization, translation |
| 23 | **Personalizer** | User Customization | Personalize responses |
| 24 | **Enhancer** | Quality Enhancement | Improve response quality |
| 25 | **Enricher** | Data Enrichment | Add supplementary information |

### Tier 6: Guardians (Agents 26-30)

Security, monitoring, and protection.

| # | Agent | Role | Responsibility |
|---|-------|------|----------------|
| 26 | **Sanitizer** | Input Sanitizer | Clean and validate inputs |
| 27 | **RateLimiter** | Traffic Control | Enforce rate limits |
| 28 | **Auditor** | Audit Logger | Log all operations |
| 29 | **Monitor** | Health Monitor | System health monitoring |
| 30 | **Recoverer** | Error Recovery | Handle failures, implement fallbacks |

### Tier 7: Architects (Agents 31-35)

Structure, design, and documentation.

| # | Agent | Role | Responsibility |
|---|-------|------|----------------|
| 31 | **Composer** | Response Composer | Compose final responses |
| 32 | **Templater** | Template Manager | Manage prompt templates |
| 33 | **Documenter** | Documentation | Generate documentation |
| 34 | **Refactorer** | Code Improvement | Suggest code improvements |
| 35 | **Tester** | Test Generator | Generate test cases |

### Tier 8: Masters (Agents 36-40)

Meta-coordination and system-level operations.

| # | Agent | Role | Responsibility |
|---|-------|------|----------------|
| 36 | **Transcender** | Supreme Orchestrator | Route all requests, coordinate system |
| 37 | **Evolver** | System Evolution | Evolve and improve system |
| 38 | **Harmonizer** | Balance Keeper | Balance load, resolve conflicts |
| 39 | **Federator** | Distributed Coordinator | Coordinate distributed operations |
| 40 | **Consciousness** | Meta-Awareness | System self-awareness, introspection |

---

## Agent Communication Bus

### Overview

The Agent Communication Bus (ACB) enables real-time inter-agent communication.

```rust
pub struct AgentBus {
    /// Channels for direct agent-to-agent communication
    pub channels: HashMap<AgentId, Sender<Message>>,

    /// Topic-based pub/sub subscriptions
    pub subscribers: HashMap<Topic, Vec<AgentId>>,

    /// Ring buffer for message logging
    pub message_log: RingBuffer<Message>,

    /// Metrics collector
    pub metrics: BusMetrics,
}
```

### Message Types

```rust
pub enum Message {
    /// Direct request to a specific agent
    Request {
        id: MessageId,
        from: AgentId,
        to: AgentId,
        payload: Value,
        timeout: Duration,
    },

    /// Response to a request
    Response {
        request_id: MessageId,
        from: AgentId,
        to: AgentId,
        payload: Result<Value, Error>,
    },

    /// Broadcast to all subscribers of a topic
    Broadcast {
        from: AgentId,
        topic: Topic,
        payload: Value,
    },

    /// Event notification
    Event {
        source: AgentId,
        event_type: EventType,
        data: Value,
    },

    /// Health check
    Heartbeat {
        agent: AgentId,
        status: HealthStatus,
        timestamp: Instant,
    },
}
```

### Topics

| Topic Pattern | Description |
|---------------|-------------|
| `agent.*` | Agent-specific events |
| `query.*` | Query processing events |
| `system.*` | System-wide events |
| `metrics.*` | Performance metrics |
| `error.*` | Error notifications |

### Participant Trait

```rust
pub trait BusParticipant: Send + Sync {
    /// Get unique agent ID
    fn agent_id(&self) -> AgentId;

    /// Subscribe to topics
    fn subscribed_topics(&self) -> Vec<Topic>;

    /// Handle incoming message
    async fn handle_message(&self, msg: Message) -> Option<Message>;

    /// Health check
    fn health_status(&self) -> HealthStatus;
}
```

---

## Memory Layer

### Architecture

```rust
pub struct MemoryLayer {
    /// Short-term memory (session cache)
    pub short_term: LruCache<String, Value>,

    /// Long-term persistent storage
    pub long_term: RocksDb,

    /// Semantic vector storage
    pub semantic: VectorStore,

    /// Time-series event history
    pub episodic: TimeSeriesDb,
}
```

### Memory Types

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY LAYER                              │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   SHORT-TERM    │    LONG-TERM    │         SEMANTIC            │
│   (LRU Cache)   │    (RocksDB)    │      (Vector Store)         │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ • Session data  │ • Agent states  │ • Prompt embeddings         │
│ • Query cache   │ • User prefs    │ • Query embeddings          │
│ • Temp results  │ • Config data   │ • Similarity search         │
│                 │ • Metrics hist  │                             │
├─────────────────┼─────────────────┼─────────────────────────────┤
│   TTL: Minutes  │  TTL: Permanent │  TTL: Configurable          │
│   Size: ~1000   │  Size: ~10GB    │  Size: ~1M vectors          │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### Memorable Trait

```rust
pub trait Memorable {
    /// Store a value with optional TTL
    async fn remember(&self, key: &str, value: Value, ttl: Option<Duration>);

    /// Retrieve a stored value
    async fn recall(&self, key: &str) -> Option<Value>;

    /// Find similar items by semantic query
    async fn associate(&self, query: &str, k: usize) -> Vec<Memory>;

    /// Remove a stored value
    async fn forget(&self, key: &str);
}
```

---

## LLM Router

### Multi-Provider Support

```rust
pub enum LLMProvider {
    OpenAI {
        model: String,
        api_key: String,
        endpoint: String,
    },
    Anthropic {
        model: String,
        api_key: String,
    },
    Google {
        model: String,
        api_key: String,
    },
    Ollama {
        model: String,
        endpoint: String,
    },
        endpoint: String,
    },
    Custom {
        endpoint: String,
        config: Value,
    },
}
```

### Routing Strategies

```rust
pub enum RoutingStrategy {
    /// Use first available provider
    Primary,

    /// Round-robin distribution
    LoadBalance,

    /// Cheapest provider first
    CostOptimized,

    /// Best quality model first
    QualityOptimized,

    /// Balance cost and quality
    Hybrid { cost_weight: f32, quality_weight: f32 },
}
```

### Router Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM ROUTER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐              │
│  │  Strategy  │   │  Fallback  │   │    Cost    │              │
│  │  Selector  │   │   Chain    │   │  Tracker   │              │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘              │
│        │                │                │                      │
│        └────────────────┼────────────────┘                      │
│                         ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Provider Pool                         │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │  Gemini  │ │  OpenAI  │ │ Anthropic│ │  Ollama  │   │    │
│  │  │ (Primary)│ │ (Backup) │ │ (Backup) │ │ (Local)  │   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Request Flow

### Query Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        QUERY PROCESSING PIPELINE                      │
└──────────────────────────────────────────────────────────────────────┘

     USER                                                        RESPONSE
       │                                                              ▲
       │ "How do I implement OAuth2?"                                 │
       ▼                                                              │
┌──────────────┐                                              ┌───────┴──────┐
│ 1. GATEWAY   │                                              │ 10. OUTPUT   │
│ Transcender  │                                              │    Router    │
└──────┬───────┘                                              └───────▲──────┘
       │                                                              │
       ▼                                                              │
┌──────────────┐                                              ┌───────┴──────┐
│ 2. SANITIZE  │                                              │ 9. COMPOSE   │
│  Sanitizer   │                                              │   Composer   │
└──────┬───────┘                                              └───────▲──────┘
       │                                                              │
       ▼                                                              │
┌──────────────┐                                              ┌───────┴──────┐
│ 3. ANALYZE   │                                              │ 8. VALIDATE  │
│  Predictor   │                                              │  Validator   │
└──────┬───────┘                                              └───────▲──────┘
       │                                                              │
       │ Intent: security, complexity: 6                              │
       ▼                                                              │
┌──────────────┐                                              ┌───────┴──────┐
│ 4. SEARCH    │                                              │ 7. CORRECT   │
│   Search     │                                              │  Corrector   │
└──────┬───────┘                                              └───────▲──────┘
       │                                                              │
       │ Found: auth-001, security-002                                │
       ▼                                                              │
┌──────────────┐          ┌──────────────┐                   ┌───────┴──────┐
│ 5. SELECT    │          │ 5b. CONTEXT  │                   │ 6. CALL LLM  │
│  Recommender │───────▶  │ Contextualizer│──────────────▶  │  LLM Router  │
└──────────────┘          └──────────────┘                   └──────────────┘
       │
       │ Strategy: Focused
       ▼
┌──────────────┐
│ 5a. SYNTH    │
│ Synthesizer  │
└──────────────┘
```

### Step Details

| Step | Agent | Action | Output |
|------|-------|--------|--------|
| 1 | Transcender | Receive query, initiate pipeline | Query object |
| 2 | Sanitizer | Validate input, clean content | Sanitized query |
| 3 | Predictor | Analyze intent, complexity | Intent analysis |
| 4 | Search | Find relevant prompts | Prompt matches |
| 5 | Recommender | Select strategy | Strategy decision |
| 5a | Synthesizer | Build meta-prompt | Meta-prompt |
| 5b | Contextualizer | Add context | Enhanced context |
| 6 | LLM Router | Call LLM provider | Raw response |
| 7 | Corrector | Detect issues | Corrections |
| 8 | Validator | Spinoza validation | Validation scores |
| 9 | Composer | Format response | Final response |
| 10 | OutputRouter | Deliver | User response |

---

## Spinoza Validation

### Philosophical Principles

Based on Baruch Spinoza's *Ethics* (1677):

| Principle | Latin | Description | Keywords |
|-----------|-------|-------------|----------|
| **CONATUS** | Self-preservation | Growth, learning, positive development | grow, learn, create, nurture |
| **RATIO** | Reason | Logical consistency, sound reasoning | therefore, because, thus |
| **LAETITIA** | Joy | Positive affect, inspiration | hope, inspire, achieve, joy |

### Validation Process

```rust
pub struct SpinozaValidator {
    conatus_keywords: Vec<String>,
    ratio_patterns: Vec<Regex>,
    laetitia_terms: Vec<String>,
    thresholds: ValidationThresholds,
}

impl SpinozaValidator {
    pub async fn validate(&self, content: &str) -> ValidationResult {
        let conatus = self.measure_conatus(content);
        let ratio = self.measure_ratio(content);
        let laetitia = self.measure_laetitia(content);

        ValidationResult {
            scores: SpinozaScores { conatus, ratio, laetitia },
            is_valid: conatus >= self.thresholds.conatus
                && ratio >= self.thresholds.ratio
                && laetitia >= self.thresholds.laetitia,
            recommendations: self.generate_recommendations(conatus, ratio, laetitia),
        }
    }
}
```

### Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.8 - 1.0 | Excellent | Pass |
| 0.6 - 0.79 | Good | Pass with notes |
| 0.4 - 0.59 | Fair | Consider correction |
| 0.2 - 0.39 | Poor | Requires correction |
| 0.0 - 0.19 | Failing | Reject or rewrite |

---

## Infrastructure

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   STANDALONE    │    │     DOCKER      │                     │
│  │   (Binary)      │    │   (Container)   │                     │
│  │                 │    │                 │                     │
│  │ panpsychism     │    │ panpsychism:1.0 │                     │
│  │   serve         │    │                 │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │            KUBERNETES (HA)                │                   │
│  │                                           │                   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │                   │
│  │  │ Pod #1  │ │ Pod #2  │ │ Pod #3  │    │                   │
│  │  └─────────┘ └─────────┘ └─────────┘    │                   │
│  │         ▼         ▼         ▼            │                   │
│  │  ┌─────────────────────────────────┐    │                   │
│  │  │        Load Balancer            │    │                   │
│  │  └─────────────────────────────────┘    │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Agent Bus | 0.5 core | 256MB | - |
| Memory Layer | 0.5 core | 1GB | 10GB |
| LLM Router | 0.25 core | 128MB | - |
| Metrics | 0.25 core | 128MB | 1GB |
| **Total (min)** | **1.5 cores** | **1.5GB** | **11GB** |

### Metrics & Monitoring

```
┌─────────────────────────────────────────────────────────────────┐
│                     OBSERVABILITY STACK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Prometheus │───▶│   Grafana   │───▶│   Alerts    │         │
│  │  (Metrics)  │    │ (Dashboard) │    │ (PagerDuty) │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         ▲                                                        │
│         │                                                        │
│  ┌─────────────┐    ┌─────────────┐                             │
│  │   Tracing   │    │   Logging   │                             │
│  │   (Jaeger)  │    │   (Loki)    │                             │
│  └─────────────┘    └─────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `panpsychism_queries_total` | Total queries processed | - |
| `panpsychism_query_latency_seconds` | Query latency histogram | p99 < 3s |
| `panpsychism_agent_invocations_total` | Agent invocation counts | - |
| `panpsychism_llm_tokens_total` | LLM tokens consumed | - |
| `panpsychism_validation_score` | Average Spinoza scores | > 0.7 |

---

## Summary

Panpsychism's architecture is designed for:

1. **Scalability**: Modular agents, stateless design
2. **Reliability**: Fallback chains, health monitoring
3. **Performance**: Caching, parallel execution
4. **Philosophy**: Spinoza validation ensures quality
5. **Flexibility**: Multi-provider LLM support

---

*For API details, see [API Reference](API_REFERENCE.md). For CLI usage, see [CLI Reference](CLI_REFERENCE.md).*
