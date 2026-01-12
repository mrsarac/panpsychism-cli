# Panpsychism API Reference

> REST API documentation for Panpsychism v4.0

This document describes the REST API endpoints for integrating Panpsychism into your applications.

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Endpoints](#endpoints)
5. [WebSocket API](#websocket-api)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Examples](#examples)

---

## Overview

The Panpsychism API provides HTTP endpoints for:

- **Query Processing**: Send queries and receive orchestrated responses
- **Agent Management**: List, invoke, and monitor agents
- **Prompt Library**: Search and manage prompts
- **System Health**: Monitor system status and metrics

### API Features

- RESTful design following OpenAPI 3.0 specification
- JSON request/response format
- JWT authentication
- WebSocket support for streaming responses
- Rate limiting with configurable thresholds

---

## Authentication

### JWT Authentication

Most endpoints require JWT authentication:

```bash
# Include token in Authorization header
curl -H "Authorization: Bearer <your-jwt-token>" \
     http://localhost:8080/api/v1/query
```

### Obtaining a Token

```bash
# Login to get token
curl -X POST http://localhost:8080/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"api_key": "your-api-key"}'
```

Response:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

### API Key Authentication (Alternative)

For simpler setups, use API key directly:

```bash
curl -H "X-API-Key: your-api-key" \
     http://localhost:8080/api/v1/query
```

---

## Base URL

| Environment | Base URL |
|-------------|----------|
| Local Development | `http://localhost:8080` |
| Docker | `http://panpsychism:8080` |
| Production | Configure in deployment |

All endpoints are prefixed with `/api/v1`.

---

## Endpoints

### Query Endpoints

#### POST /api/v1/query

Process a query through the orchestration pipeline.

**Request:**

```bash
curl -X POST http://localhost:8080/api/v1/query \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How do I implement OAuth2 authentication?",
       "options": {
         "strategy": "auto",
         "verbose": false,
         "max_tokens": 2048,
         "temperature": 0.7
       }
     }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | The user's query |
| `options.strategy` | string | No | `auto`, `focused`, `ensemble`, `chain`, `parallel` |
| `options.verbose` | boolean | No | Include reasoning trace |
| `options.max_tokens` | integer | No | Maximum response tokens |
| `options.temperature` | float | No | LLM temperature (0.0-1.0) |
| `options.agent` | string | No | Specific agent to use |

**Response:**

```json
{
  "id": "qry_abc123",
  "query": "How do I implement OAuth2 authentication?",
  "response": "To implement OAuth2 authentication, follow these steps...",
  "metadata": {
    "strategy": "focused",
    "prompts_used": ["auth-001", "security-002"],
    "agents_invoked": ["synthesizer", "validator"],
    "latency_ms": 1234,
    "tokens_used": 512
  },
  "validation": {
    "conatus": 0.85,
    "ratio": 0.92,
    "laetitia": 0.78,
    "passed": true
  },
  "created_at": "2026-01-09T10:30:00Z"
}
```

---

#### POST /api/v1/query/stream

Process a query with streaming response.

**Request:**

```bash
curl -X POST http://localhost:8080/api/v1/query/stream \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -d '{
       "query": "Explain quantum computing",
       "options": {}
     }'
```

**Response (Server-Sent Events):**

```
event: start
data: {"id": "qry_abc123", "status": "processing"}

event: token
data: {"token": "Quantum"}

event: token
data: {"token": " computing"}

event: token
data: {"token": " is"}

...

event: complete
data: {"id": "qry_abc123", "status": "complete", "tokens_used": 256}
```

---

### Agent Endpoints

#### GET /api/v1/agents

List all available agents.

**Request:**

```bash
curl http://localhost:8080/api/v1/agents \
     -H "Authorization: Bearer <token>"
```

**Response:**

```json
{
  "agents": [
    {
      "id": "transcender",
      "name": "Transcender",
      "tier": 8,
      "description": "Supreme Orchestrator & Router",
      "status": "active"
    },
    {
      "id": "synthesizer",
      "name": "Synthesizer",
      "tier": 3,
      "description": "Meta-Prompt Builder",
      "status": "active"
    },
    ...
  ],
  "total": 40,
  "active": 40
}
```

---

#### GET /api/v1/agents/{agent_id}

Get details of a specific agent.

**Request:**

```bash
curl http://localhost:8080/api/v1/agents/synthesizer \
     -H "Authorization: Bearer <token>"
```

**Response:**

```json
{
  "id": "synthesizer",
  "name": "Synthesizer",
  "tier": 3,
  "description": "Meta-Prompt Builder",
  "status": "active",
  "capabilities": [
    "prompt_synthesis",
    "context_building",
    "meta_prompt_generation"
  ],
  "metrics": {
    "invocations": 1234,
    "avg_latency_ms": 45,
    "success_rate": 0.99
  }
}
```

---

#### POST /api/v1/agents/{agent_id}/invoke

Invoke a specific agent directly.

**Request:**

```bash
curl -X POST http://localhost:8080/api/v1/agents/synthesizer/invoke \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{
       "input": "Create a code review prompt",
       "context": {
         "language": "rust",
         "focus": "security"
       }
     }'
```

**Response:**

```json
{
  "agent_id": "synthesizer",
  "output": "You are a senior Rust developer...",
  "metadata": {
    "latency_ms": 23,
    "tokens_used": 128
  }
}
```

---

#### GET /api/v1/agents/{agent_id}/status

Get the current status of an agent.

**Request:**

```bash
curl http://localhost:8080/api/v1/agents/synthesizer/status \
     -H "Authorization: Bearer <token>"
```

**Response:**

```json
{
  "agent_id": "synthesizer",
  "status": "active",
  "health": "healthy",
  "uptime_seconds": 86400,
  "current_load": 0.15,
  "queue_depth": 2
}
```

---

### Prompt Endpoints

#### GET /api/v1/prompts

List all prompts in the library.

**Request:**

```bash
curl http://localhost:8080/api/v1/prompts \
     -H "Authorization: Bearer <token>"
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Filter by category |
| `tags` | string | Comma-separated tags |
| `limit` | integer | Results per page (default: 20) |
| `offset` | integer | Pagination offset |

**Response:**

```json
{
  "prompts": [
    {
      "id": "auth-001",
      "title": "OAuth2 Authentication Guide",
      "category": "security",
      "tags": ["authentication", "oauth", "jwt"],
      "version": "1.0.0",
      "privacy_tier": "public"
    },
    ...
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

---

#### GET /api/v1/prompts/{prompt_id}

Get a specific prompt.

**Request:**

```bash
curl http://localhost:8080/api/v1/prompts/auth-001 \
     -H "Authorization: Bearer <token>"
```

**Response:**

```json
{
  "id": "auth-001",
  "title": "OAuth2 Authentication Guide",
  "description": "Comprehensive guide for implementing OAuth2",
  "category": "security",
  "tags": ["authentication", "oauth", "jwt"],
  "version": "1.0.0",
  "author": "mustafa",
  "privacy_tier": "public",
  "content": "# OAuth2 Authentication Guide\n\nThis prompt helps implement...",
  "created_at": "2026-01-01T00:00:00Z",
  "updated_at": "2026-01-09T10:00:00Z"
}
```

---

#### POST /api/v1/prompts/search

Search for prompts.

**Request:**

```bash
curl -X POST http://localhost:8080/api/v1/prompts/search \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "authentication security",
       "top_k": 5,
       "min_score": 0.1
     }'
```

**Response:**

```json
{
  "results": [
    {
      "prompt": {
        "id": "auth-001",
        "title": "OAuth2 Authentication Guide",
        "category": "security"
      },
      "score": 0.87,
      "highlights": ["authentication", "security"]
    },
    ...
  ],
  "total": 3,
  "latency_ms": 12
}
```

---

### Health Endpoints

#### GET /api/v1/health

Basic health check.

**Request:**

```bash
curl http://localhost:8080/api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400
}
```

---

#### GET /api/v1/health/detailed

Detailed health check with component status.

**Request:**

```bash
curl http://localhost:8080/api/v1/health/detailed \
     -H "Authorization: Bearer <token>"
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "components": {
    "agent_bus": {"status": "healthy", "latency_ms": 1},
    "memory_layer": {"status": "healthy", "usage_mb": 128},
    "llm_router": {"status": "healthy", "active_providers": 3},
    "prompt_store": {"status": "healthy", "prompt_count": 42}
  },
  "agents": {
    "total": 40,
    "active": 40,
    "degraded": 0
  }
}
```

---

#### GET /api/v1/metrics

Prometheus-compatible metrics.

**Request:**

```bash
curl http://localhost:8080/api/v1/metrics
```

**Response:**

```
# HELP panpsychism_queries_total Total queries processed
# TYPE panpsychism_queries_total counter
panpsychism_queries_total{status="success"} 1234
panpsychism_queries_total{status="error"} 12

# HELP panpsychism_query_latency_seconds Query latency histogram
# TYPE panpsychism_query_latency_seconds histogram
panpsychism_query_latency_seconds_bucket{le="0.1"} 100
panpsychism_query_latency_seconds_bucket{le="0.5"} 500
panpsychism_query_latency_seconds_bucket{le="1.0"} 800

# HELP panpsychism_agent_invocations_total Agent invocation counts
# TYPE panpsychism_agent_invocations_total counter
panpsychism_agent_invocations_total{agent="synthesizer"} 1234
panpsychism_agent_invocations_total{agent="validator"} 1234
```

---

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/api/v1/stream');

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-jwt-token'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

### Message Types

#### Client → Server

**Query Request:**
```json
{
  "type": "query",
  "id": "req_123",
  "query": "How do I implement auth?",
  "options": {}
}
```

**Subscribe to Agent Events:**
```json
{
  "type": "subscribe",
  "topics": ["agent.synthesizer", "agent.validator"]
}
```

#### Server → Client

**Token Stream:**
```json
{
  "type": "token",
  "request_id": "req_123",
  "token": "To implement"
}
```

**Agent Event:**
```json
{
  "type": "event",
  "topic": "agent.synthesizer",
  "data": {
    "action": "invoked",
    "latency_ms": 45
  }
}
```

**Query Complete:**
```json
{
  "type": "complete",
  "request_id": "req_123",
  "response": "Full response here...",
  "metadata": {}
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Query cannot be empty",
    "details": {
      "field": "query",
      "constraint": "required"
    }
  },
  "request_id": "req_abc123"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `LLM_ERROR` | 502 | LLM provider error |
| `TIMEOUT` | 504 | Request timeout |

---

## Rate Limiting

### Default Limits

| Tier | Requests/Minute | Tokens/Day |
|------|-----------------|------------|
| Free | 10 | 10,000 |
| Pro | 100 | 100,000 |
| Enterprise | 1,000 | Unlimited |

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704790800
```

### Handling Rate Limits

```javascript
if (response.status === 429) {
  const resetTime = response.headers.get('X-RateLimit-Reset');
  const waitMs = (resetTime * 1000) - Date.now();
  await sleep(waitMs);
  // Retry request
}
```

---

## Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8080/api/v1"
API_KEY = "your-api-key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Query
response = requests.post(
    f"{BASE_URL}/query",
    headers=headers,
    json={
        "query": "How do I implement OAuth2?",
        "options": {"strategy": "auto"}
    }
)

result = response.json()
print(result["response"])
```

### JavaScript/TypeScript Client

```typescript
const BASE_URL = "http://localhost:8080/api/v1";

async function query(text: string): Promise<string> {
  const response = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: text,
      options: { strategy: "auto" }
    }),
  });

  const data = await response.json();
  return data.response;
}

// Usage
const answer = await query("How do I implement OAuth2?");
console.log(answer);
```

### cURL Examples

```bash
# Basic query
curl -X POST http://localhost:8080/api/v1/query \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain REST APIs"}'

# Search prompts
curl "http://localhost:8080/api/v1/prompts/search" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "authentication", "top_k": 5}'

# Check health
curl http://localhost:8080/api/v1/health

# List agents
curl http://localhost:8080/api/v1/agents \
     -H "Authorization: Bearer $TOKEN"
```

---

*For CLI usage, see [CLI Reference](CLI_REFERENCE.md). For architecture details, see [Architecture](ARCHITECTURE.md).*
