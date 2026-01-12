# Panpsychism v2.0 - Oracle Tier Implementation Plan

> **Status:** AWAITING APPROVAL
> **Date:** 2026-01-08
> **Pattern:** Ralph Wiggum (RW-5)
> **Method:** BD Task Tracking

---

## Executive Summary

Panpsychism v2.0, Sorcerer's Guild mimarisinin son katmanı olan **Tier 4: Oracles** (Agent 16-20) implementasyonunu içerir. Bu ajanlar, sistemin tahmin, öneri, değerlendirme, hata ayıklama ve öğrenme yeteneklerini sağlar.

### Version History

| Version | Tier | Agents | Status |
|---------|------|--------|--------|
| v1.0.0 | Tier 1-2 | 1-10 | ✅ Complete |
| v1.5.0 | Tier 3 | 11-15 | ✅ Complete |
| **v2.0.0** | **Tier 4** | **16-20** | 🎯 **PLANNING** |

---

## Tier 4: The Oracles (Agents 16-20)

### Agent Specifications

#### Agent 16: Predictor (Future Seer) 🔮
**Role:** Anticipate user needs and predict next queries

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Query prediction, intent anticipation, trend analysis |
| **Input** | Query history, session context, user patterns |
| **Output** | PredictionResult with confidence scores |
| **Dependencies** | Contextualizer, Search, Analyzer |

**Core Types:**
```rust
pub struct PredictorAgent { /* ... */ }
pub struct Prediction {
    pub predicted_query: String,
    pub confidence: f64,
    pub reasoning: String,
    pub alternatives: Vec<String>,
}
pub enum PredictionStrategy {
    PatternBased,      // Historical pattern matching
    ContextualFlow,    // Session context analysis
    SemanticChain,     // Semantic similarity chains
    HybridEnsemble,    // Combined strategies
}
```

**Test Requirements:** ~40 unit tests
- Pattern prediction accuracy
- Context-based predictions
- Confidence calibration
- Multi-step prediction chains

---

#### Agent 17: Recommender (Path Advisor) 🧭
**Role:** Suggest relevant prompts and optimal paths

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Prompt recommendation, path optimization, suggestion ranking |
| **Input** | Current query, user history, available prompts |
| **Output** | RecommendationSet with ranked suggestions |
| **Dependencies** | Search, Analyzer, Predictor |

**Core Types:**
```rust
pub struct RecommenderAgent { /* ... */ }
pub struct Recommendation {
    pub prompt_id: PromptId,
    pub relevance_score: f64,
    pub explanation: String,
    pub tags: Vec<String>,
}
pub struct RecommendationConfig {
    pub max_recommendations: usize,
    pub diversity_factor: f64,
    pub recency_weight: f64,
    pub popularity_weight: f64,
}
pub enum RecommendationStrategy {
    ContentBased,       // Similar content
    Collaborative,      // User behavior patterns
    KnowledgeBased,     // Domain rules
    Hybrid,             // Combined approach
}
```

**Test Requirements:** ~45 unit tests
- Relevance ranking accuracy
- Diversity in recommendations
- Cold start handling
- Explanation generation

---

#### Agent 18: Evaluator (Quality Oracle) ⚖️
**Role:** Assess response quality and provide feedback

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Quality assessment, scoring, improvement suggestions |
| **Input** | Generated response, original query, context |
| **Output** | EvaluationReport with detailed metrics |
| **Dependencies** | Validator, Analyzer, Summarizer |

**Core Types:**
```rust
pub struct EvaluatorAgent { /* ... */ }
pub struct EvaluationReport {
    pub overall_score: f64,
    pub dimensions: EvaluationDimensions,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub suggestions: Vec<Improvement>,
}
pub struct EvaluationDimensions {
    pub relevance: f64,      // Query-response alignment
    pub completeness: f64,   // Coverage of topic
    pub clarity: f64,        // Readability and structure
    pub accuracy: f64,       // Factual correctness signals
    pub actionability: f64,  // Practical usefulness
}
pub enum EvaluationLevel {
    Quick,      // Fast heuristic check
    Standard,   // Balanced analysis
    Deep,       // Comprehensive evaluation
}
```

**Test Requirements:** ~50 unit tests
- Dimension scoring accuracy
- Threshold calibration
- Suggestion relevance
- Multi-level evaluation

---

#### Agent 19: Debugger (Error Hunter) 🐛
**Role:** Identify issues and suggest fixes

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Error detection, root cause analysis, fix suggestions |
| **Input** | Failed operation, error context, system state |
| **Output** | DebugReport with fixes |
| **Dependencies** | Corrector, Analyzer, Validator |

**Core Types:**
```rust
pub struct DebuggerAgent { /* ... */ }
pub struct DebugReport {
    pub issue: Issue,
    pub root_cause: RootCause,
    pub suggested_fixes: Vec<Fix>,
    pub prevention_tips: Vec<String>,
}
pub struct Issue {
    pub category: IssueCategory,
    pub severity: Severity,
    pub description: String,
    pub location: Option<String>,
}
pub enum IssueCategory {
    QueryMalformed,      // Bad input
    PromptNotFound,      // Missing prompt
    ValidationFailed,    // Spinoza rejection
    SynthesisFailed,     // Generation error
    TimeoutExceeded,     // Performance issue
    ResourceExhausted,   // Limits hit
}
pub enum Severity {
    Critical,   // System cannot proceed
    High,       // Major functionality affected
    Medium,     // Degraded experience
    Low,        // Minor inconvenience
}
```

**Test Requirements:** ~45 unit tests
- Issue categorization
- Root cause accuracy
- Fix effectiveness
- Severity assessment

---

#### Agent 20: Learner (Knowledge Absorber) 📚
**Role:** Learn from interactions and improve over time

| Aspect | Specification |
|--------|---------------|
| **Purpose** | Pattern learning, knowledge accumulation, self-improvement |
| **Input** | Interaction history, feedback, outcomes |
| **Output** | LearningInsights, updated knowledge base |
| **Dependencies** | All agents (meta-learning) |

**Core Types:**
```rust
pub struct LearnerAgent { /* ... */ }
pub struct LearningInsights {
    pub patterns_discovered: Vec<Pattern>,
    pub knowledge_updates: Vec<KnowledgeUpdate>,
    pub performance_trends: PerformanceTrends,
    pub recommendations: Vec<SystemRecommendation>,
}
pub struct Pattern {
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub occurrences: usize,
    pub description: String,
}
pub enum PatternType {
    QueryPattern,       // Common query structures
    ErrorPattern,       // Recurring issues
    SuccessPattern,     // Effective approaches
    UserBehavior,       // Usage patterns
}
pub struct KnowledgeUpdate {
    pub category: String,
    pub insight: String,
    pub evidence_count: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

**Test Requirements:** ~40 unit tests
- Pattern detection accuracy
- Knowledge persistence
- Trend analysis
- Self-improvement metrics

---

## Ralph Wiggum Execution Protocol

### Constraints (RW-5)

| Parameter | Value |
|-----------|-------|
| Max Parallel Agents | **5** |
| Max Iterations per Agent | **5** |
| Completion Promise | "DONE" |
| User Approval | **REQUIRED** |

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              RALPH WIGGUM WAVE 2 (v2.0 Oracles)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. USER APPROVAL ────▶ "Onaylıyorum, v2.0 başlasın"           │
│         │                                                       │
│         ▼                                                       │
│  2. BD TASKS ────────▶ bd create (5 agent tasks)               │
│         │                                                       │
│         ▼                                                       │
│  3. GIT BRANCH ──────▶ git checkout -b feature/v2.0-oracles    │
│         │                                                       │
│         ▼                                                       │
│  4. SPAWN 5 AGENTS ──▶ Parallel Task tool execution            │
│         │              ├── Agent 16: Predictor                  │
│         │              ├── Agent 17: Recommender                │
│         │              ├── Agent 18: Evaluator                  │
│         │              ├── Agent 19: Debugger                   │
│         │              └── Agent 20: Learner                    │
│         │                                                       │
│         ▼                                                       │
│  5. MERGE & VERIFY ──▶ cargo check + cargo test                │
│         │                                                       │
│         ▼                                                       │
│  6. BD CLOSE ────────▶ Close all 5 agent tasks                 │
│         │                                                       │
│         ▼                                                       │
│  7. RELEASE ─────────▶ git tag v2.0.0 + gh release             │
│                                                                 │
│  ⚠️ AGENTS WILL NOT START UNTIL USER SAYS:                     │
│     "Onaylıyorum, v2.0 başlasın"                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Prompt Templates

### Agent 16: Predictor Prompt
```
You are implementing Agent 16: Predictor (Future Seer) for Panpsychism v2.0.

ROLE: Predict user's next queries based on context and patterns.

CREATE FILE: src/predictor.rs

REQUIREMENTS:
1. PredictorAgent struct with builder pattern
2. Prediction struct with confidence scores
3. PredictionStrategy enum (PatternBased, ContextualFlow, SemanticChain, HybridEnsemble)
4. PredictorConfig for customization
5. Async predict() method
6. Integration with Contextualizer for session history
7. ~40 unit tests

PATTERNS TO FOLLOW:
- Match existing agent patterns (see contextualizer.rs, summarizer.rs)
- Use crate::Result for error handling
- Implement Debug, Clone where appropriate
- Document all public items

UPDATE lib.rs:
- Add `pub mod predictor;`
- Add re-exports for public types

OUTPUT: Complete predictor.rs file with tests
SAY "DONE" when complete.
```

### Agent 17: Recommender Prompt
```
You are implementing Agent 17: Recommender (Path Advisor) for Panpsychism v2.0.

ROLE: Suggest relevant prompts and optimal paths to users.

CREATE FILE: src/recommender.rs

REQUIREMENTS:
1. RecommenderAgent struct with builder pattern
2. Recommendation struct with relevance scoring
3. RecommendationStrategy enum (ContentBased, Collaborative, KnowledgeBased, Hybrid)
4. RecommendationConfig with diversity/recency/popularity weights
5. Async recommend() method returning ranked list
6. Explanation generation for each recommendation
7. ~45 unit tests

PATTERNS TO FOLLOW:
- Match existing agent patterns (see expander.rs, formatter.rs)
- Use crate::Result for error handling
- Score normalization to 0.0-1.0 range
- Document all public items

UPDATE lib.rs:
- Add `pub mod recommender;`
- Add re-exports for public types

OUTPUT: Complete recommender.rs file with tests
SAY "DONE" when complete.
```

### Agent 18: Evaluator Prompt
```
You are implementing Agent 18: Evaluator (Quality Oracle) for Panpsychism v2.0.

ROLE: Assess response quality across multiple dimensions.

CREATE FILE: src/evaluator.rs

REQUIREMENTS:
1. EvaluatorAgent struct with builder pattern
2. EvaluationReport with overall score and dimensions
3. EvaluationDimensions (relevance, completeness, clarity, accuracy, actionability)
4. EvaluationLevel enum (Quick, Standard, Deep)
5. Improvement suggestions with priority
6. Async evaluate() method
7. ~50 unit tests

PATTERNS TO FOLLOW:
- Match validator.rs pattern for scoring
- Use crate::Result for error handling
- Spinoza alignment (CONATUS/RATIO/LAETITIA consideration)
- Document all public items

UPDATE lib.rs:
- Add `pub mod evaluator;`
- Add re-exports for public types

OUTPUT: Complete evaluator.rs file with tests
SAY "DONE" when complete.
```

### Agent 19: Debugger Prompt
```
You are implementing Agent 19: Debugger (Error Hunter) for Panpsychism v2.0.

ROLE: Identify issues, analyze root causes, suggest fixes.

CREATE FILE: src/debugger.rs

REQUIREMENTS:
1. DebuggerAgent struct with builder pattern
2. DebugReport with issue, root cause, and fixes
3. Issue struct with category and severity
4. IssueCategory enum (QueryMalformed, PromptNotFound, ValidationFailed, etc.)
5. Severity enum (Critical, High, Medium, Low)
6. Fix struct with steps and confidence
7. Async debug() method
8. ~45 unit tests

PATTERNS TO FOLLOW:
- Match corrector.rs pattern for issue handling
- Use crate::Result and crate::Error types
- Integrate with existing error types
- Document all public items

UPDATE lib.rs:
- Add `pub mod debugger;`
- Add re-exports for public types

OUTPUT: Complete debugger.rs file with tests
SAY "DONE" when complete.
```

### Agent 20: Learner Prompt
```
You are implementing Agent 20: Learner (Knowledge Absorber) for Panpsychism v2.0.

ROLE: Learn from interactions, discover patterns, improve system.

CREATE FILE: src/learner.rs

REQUIREMENTS:
1. LearnerAgent struct with builder pattern
2. LearningInsights with patterns and knowledge updates
3. Pattern struct with type, confidence, occurrences
4. PatternType enum (QueryPattern, ErrorPattern, SuccessPattern, UserBehavior)
5. KnowledgeUpdate for incremental learning
6. PerformanceTrends tracking
7. Async learn() and get_insights() methods
8. ~40 unit tests

PATTERNS TO FOLLOW:
- Match analyzer.rs pattern for pattern detection
- Use chrono for timestamps
- Consider persistence (in-memory for now, extensible later)
- Document all public items

UPDATE lib.rs:
- Add `pub mod learner;`
- Add re-exports for public types

OUTPUT: Complete learner.rs file with tests
SAY "DONE" when complete.
```

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Agents Complete | 5/5 | All Oracle agents functional |
| Test Coverage | 200+ new tests | cargo test passes |
| Code Quality | 0 errors | cargo check clean |
| Documentation | All public APIs | rustdoc generation |
| No Regressions | 376+ tests pass | Existing tests unchanged |

## Expected Metrics After v2.0

| Metric | v1.5.0 | v2.0.0 (Target) | Growth |
|--------|--------|-----------------|--------|
| Source Files | 20 | 25 | +5 |
| Lines of Code | 21,185 | ~28,000 | +~7,000 |
| Unit Tests | 376 | ~600 | +~220 |
| Total Agents | 15 | 20 | +5 |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Agent interdependencies | Medium | Medium | Clear interfaces, mock dependencies |
| Test overlap | Low | Low | Unique test scenarios per agent |
| lib.rs merge conflicts | Medium | Low | Sequential lib.rs updates |
| Performance regression | Low | Medium | Benchmark critical paths |

---

## BD Task Structure

```
panpsychism-v2-epic (Epic)
├── panpsychism-v2-predictor (Task) - Agent 16
├── panpsychism-v2-recommender (Task) - Agent 17
├── panpsychism-v2-evaluator (Task) - Agent 18
├── panpsychism-v2-debugger (Task) - Agent 19
└── panpsychism-v2-learner (Task) - Agent 20
```

---

## Approval Checklist

Before execution, confirm:

- [ ] Plan reviewed and understood
- [ ] 5 Oracle agents specs approved
- [ ] Ralph Wiggum RW-5 pattern accepted
- [ ] BD task structure approved
- [ ] Ready for parallel execution

---

**To approve and start v2.0 implementation, say:**

> "Onaylıyorum, v2.0 başlasın"

---

*Plan Version: 1.0*
*Created: 2026-01-08*
*Author: Claude Opus 4.5 + Strategy Board*
