# REST API Client Example

Demonstrates building a REST API server and WebSocket endpoint for Panpsychism.

## What This Example Does

1. Configure a REST API server with endpoints
2. Set up WebSocket for streaming responses
3. Add middleware (CORS, rate limiting, auth)
4. Show client usage (curl, Rust)

## Architecture

```
+------------------+
|   REST Server    |  <-- HTTP endpoints
+------------------+
         |
+------------------+
|WebSocket Server  |  <-- Real-time streaming
+------------------+
         |
+------------------+
|   Orchestrator   |  <-- Query processing
+------------------+
```

## Running

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the example
cargo run --example rest-api-client

# In a real server:
# cargo run --release --example rest-api-client
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | Process a query |
| GET | `/prompts` | List available prompts |
| GET | `/stats` | Server statistics |

## curl Examples

```bash
# Health check
curl http://127.0.0.1:3000/health

# Query
curl -X POST http://127.0.0.1:3000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I use async in Rust?"}'

# List prompts
curl http://127.0.0.1:3000/prompts
```

## WebSocket

Connect to `ws://127.0.0.1:3001` for streaming:

```bash
# Using websocat
websocat ws://127.0.0.1:3001

# Send query
{"type": "query", "data": {"query": "Explain ownership"}}

# Receive streaming response
{"type": "chunk", "data": "Ownership"}
{"type": "chunk", "data": " is"}
{"type": "chunk", "data": " Rust's"}
...
{"type": "done", "data": {"tokens": 150}}
```

## Key Concepts

### API Server

```rust
let api_server = ApiServerBuilder::new()
    .with_config(ApiConfig::default()
        .with_host("127.0.0.1")
        .with_port(3000))
    .with_router(router)
    .with_middleware(LoggingMiddleware::new())
    .with_middleware(CorsMiddleware::new())
    .build()?;

api_server.run().await?;
```

### Router

```rust
let mut router = Router::new();

router.route(Route::new(Method::Get, "/health")
    .handler(|_req| async move {
        Response::json(Status::Ok, &json!({"status": "ok"}))
    }));

router.route(Route::new(Method::Post, "/query")
    .handler(|req| async move {
        let body: QueryRequest = req.json()?;
        // Process...
        Response::json(Status::Ok, &result)
    }));
```

### Middleware

```rust
// CORS
CorsMiddleware::new()
    .allow_origin("*")
    .allow_methods(vec!["GET", "POST"])

// Rate limiting
RateLimitMiddleware::new(60, 10)  // 60 req/min, burst of 10

// Authentication
AuthMiddleware::new(AuthConfig::default()
    .with_api_key_header("X-API-Key")
    .with_api_keys(vec!["sk-live-abc123".to_string()]))

// Logging
LoggingMiddleware::new()
```

### WebSocket Server

```rust
let ws_server = WebSocketServerBuilder::new()
    .with_config(WebSocketConfig::default()
        .with_port(3001)
        .with_max_connections(100))
    .with_orchestrator(orchestrator)
    .build()?;

ws_server.run().await?;
```

## Rust Client

```rust
use reqwest::Client;

async fn query_api(query: &str) -> Result<String> {
    let client = Client::new();

    let response = client
        .post("http://127.0.0.1:3000/query")
        .json(&json!({"query": query}))
        .send()
        .await?
        .json::<QueryResponse>()
        .await?;

    Ok(response.content)
}
```

## Request/Response Format

### Query Request

```json
{
  "query": "How do I implement auth?",
  "strategy": "focused",
  "max_tokens": 500
}
```

### Query Response

```json
{
  "content": "To implement authentication...",
  "strategy": "Focused",
  "prompts_used": 3,
  "confidence": 0.87,
  "processing_time_ms": 1234
}
```

### Health Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 127.0.0.1 | API bind address |
| `API_PORT` | 3000 | API port |
| `WS_PORT` | 3001 | WebSocket port |
| `RATE_LIMIT_RPM` | 60 | Requests per minute |

### Config File

```yaml
api:
  host: 127.0.0.1
  port: 3000
  cors: true
  rate_limit:
    requests_per_minute: 60
    burst_size: 10

websocket:
  port: 3001
  max_connections: 100
  heartbeat_interval: 30
```

## Security Best Practices

1. **Always use HTTPS in production**
2. **Enable authentication for sensitive endpoints**
3. **Set appropriate rate limits**
4. **Validate all input**
5. **Log requests for auditing**

## Next Steps

- See [basic-query](../basic-query/) for core concepts
- Explore [multi-agent](../multi-agent/) for workflows
- Check [llm-router](../llm-router/) for provider routing
