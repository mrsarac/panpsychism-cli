//! Example: REST API Server
//!
//! This example demonstrates how to create a REST API server
//! for Panpsychism, enabling HTTP-based integration with other services.
//!
//! Features:
//! - REST endpoints for query processing
//! - WebSocket for real-time streaming
//! - Health checks and metrics
//! - Rate limiting and authentication

use panpsychism::{
    api::{
        // REST API
        ApiConfig, ApiServerBuilder, Method, Request, Response, Route, Router, Status,
        // Middleware
        CorsMiddleware, LoggingMiddleware, RateLimitMiddleware, RestRateLimitConfig,
        // WebSocket
        WebSocketConfig, WebSocketServerBuilder,
    },
    config::Config,
    orchestrator::Orchestrator,
    Result, VERSION,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request body for query endpoint
#[derive(Debug, Deserialize)]
struct QueryRequest {
    query: String,
    #[serde(default)]
    strategy: Option<String>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

/// Response body for query endpoint
#[derive(Debug, Serialize)]
struct QueryResponse {
    content: String,
    strategy: String,
    prompts_used: usize,
    confidence: f64,
    processing_time_ms: u64,
}

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    uptime_seconds: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    panpsychism::setup_logging();

    println!("=== Panpsychism REST API Example ===\n");

    // Load configuration
    let config = Config::load()?;

    // Initialize the orchestrator (shared across handlers)
    let orchestrator = Arc::new(Orchestrator::new());

    // === Part 1: Build the REST API ===
    println!("--- Part 1: REST API Server ---\n");

    let api_config = ApiConfig::default()
        .host("127.0.0.1")
        .port(3000)
        .cors(true)
        .rate_limit(
            RestRateLimitConfig::default()
                .requests_per_minute(60)
                .burst_size(10),
        );

    // Build the router
    let router = build_router(orchestrator.clone())?;

    // Create the API server
    let api_server = ApiServerBuilder::new()
        .config(api_config)
        .router(router)
        .middleware(LoggingMiddleware::new())
        .middleware(
            CorsMiddleware::new()
                .allow_origin("*")
                .allow_methods(vec!["GET", "POST", "OPTIONS"]),
        )
        .middleware(RateLimitMiddleware::new(60, 10))
        .build()?;

    println!("REST API configured:");
    println!("  Host: 127.0.0.1:3000");
    println!("  Endpoints:");
    println!("    GET  /health");
    println!("    POST /query");
    println!("    GET  /prompts");
    println!("    GET  /stats");

    // === Part 2: WebSocket Server ===
    println!("\n--- Part 2: WebSocket Server ---\n");

    let ws_config = WebSocketConfig::default()
        .host("127.0.0.1")
        .port(3001)
        .max_connections(100)
        .heartbeat_interval(30);

    let ws_server = WebSocketServerBuilder::new()
        .config(ws_config)
        .build()?;

    println!("WebSocket server configured:");
    println!("  Host: 127.0.0.1:3001");
    println!("  Max connections: 100");
    println!("  Heartbeat: 30s");

    // === Part 3: Start Servers ===
    println!("\n--- Part 3: Starting Servers ---\n");

    // In production, you would run these
    // For this example, we just show the configuration
    println!("To start servers (in production):");
    println!("  api_server.run().await?;");
    println!("  ws_server.run().await?;");

    // === Part 4: Example Client Usage ===
    println!("\n--- Part 4: Client Examples ---\n");

    println!("curl examples:");
    println!();
    println!("# Health check");
    println!("curl http://127.0.0.1:3000/health");
    println!();
    println!("# Query");
    println!(r#"curl -X POST http://127.0.0.1:3000/query \"#);
    println!(r#"  -H "Content-Type: application/json" \"#);
    println!(r#"  -d '{{"query": "How do I use async in Rust?"}}'"#);
    println!();
    println!("# List prompts");
    println!("curl http://127.0.0.1:3000/prompts");
    println!();
    println!("# WebSocket");
    println!("websocat ws://127.0.0.1:3001");

    // === Part 5: Programmatic Client ===
    println!("\n--- Part 5: Rust Client ---\n");

    // Example of making a request programmatically
    demonstrate_client();

    Ok(())
}

/// Build the API router with all endpoints
fn build_router(orchestrator: Arc<Orchestrator>) -> Result<Router> {
    let mut router = Router::new();

    // GET /health - Health check
    router.route(
        Route::new(Method::Get, "/health").handler(|_req: Request| async move {
            let response = HealthResponse {
                status: "healthy".to_string(),
                version: VERSION.to_string(),
                uptime_seconds: 0, // Would be calculated in production
            };
            Response::json(Status::Ok, &response)
        }),
    );

    // POST /query - Process a query
    let orch = orchestrator.clone();
    router.route(
        Route::new(Method::Post, "/query").handler(move |req: Request| {
            let orch = orch.clone();
            async move {
                // Parse request body
                let body: QueryRequest = match req.json() {
                    Ok(b) => b,
                    Err(e) => return Response::error(Status::BadRequest, &e.to_string()),
                };

                // Process the query
                let start = std::time::Instant::now();
                match orch.analyze_intent(&body.query).await {
                    Ok(intent) => {
                        let response = QueryResponse {
                            content: format!(
                                "Analysis complete. Category: {:?}, Strategy: {:?}",
                                intent.category, intent.strategy
                            ),
                            strategy: format!("{:?}", intent.strategy),
                            prompts_used: 0, // Would be filled from actual processing
                            confidence: 0.85,
                            processing_time_ms: start.elapsed().as_millis() as u64,
                        };
                        Response::json(Status::Ok, &response)
                    }
                    Err(e) => Response::error(Status::InternalServerError, &e.to_string()),
                }
            }
        }),
    );

    // GET /prompts - List available prompts
    router.route(
        Route::new(Method::Get, "/prompts").handler(|_req: Request| async move {
            // Would return list of prompts from store
            Response::json(
                Status::Ok,
                &serde_json::json!({
                    "prompts": [
                        {"id": "code-review", "category": "Technical"},
                        {"id": "explain-concept", "category": "Education"},
                        {"id": "debug-error", "category": "Debugging"},
                    ]
                }),
            )
        }),
    );

    // GET /stats - Server statistics
    router.route(
        Route::new(Method::Get, "/stats").handler(|_req: Request| async move {
            Response::json(
                Status::Ok,
                &serde_json::json!({
                    "total_requests": 0,
                    "avg_response_time_ms": 0,
                    "active_connections": 0,
                }),
            )
        }),
    );

    Ok(router)
}

/// Demonstrate making requests with a Rust client
fn demonstrate_client() {
    println!("Rust client example:");
    println!();
    println!(
        r#"
use reqwest::Client;

#[derive(Serialize)]
struct QueryRequest {{
    query: String,
}}

#[derive(Deserialize)]
struct QueryResponse {{
    content: String,
    confidence: f64,
}}

async fn query_api(query: &str) -> Result<QueryResponse> {{
    let client = Client::new();

    let response = client
        .post("http://127.0.0.1:3000/query")
        .json(&QueryRequest {{ query: query.to_string() }})
        .send()
        .await?
        .json::<QueryResponse>()
        .await?;

    Ok(response)
}}

// Usage
let result = query_api("Explain ownership in Rust").await?;
println!("Response: {{}}", result.content);
"#
    );
}

/// Example: WebSocket message handler
#[allow(dead_code)]
fn websocket_handler_docs() {
    println!("WebSocket message types:");
    println!("  - query: Process a query and stream response");
    println!("  - subscribe: Subscribe to updates");
    println!("  - ping: Keep connection alive");
    println!();
    println!(
        r#"
// Client sends:
{{"type": "query", "data": {{"query": "Hello"}}}}

// Server streams back:
{{"type": "chunk", "data": "Hello"}}
{{"type": "chunk", "data": "!"}}
{{"type": "done", "data": {{"tokens": 2}}}}
"#
    );
}

/// Example: With authentication middleware
#[allow(dead_code)]
fn authenticated_server_example() -> Result<()> {
    use panpsychism::api::{AuthConfig, AuthMiddleware};

    let auth_config = AuthConfig::default()
        .api_key_header("X-API-Key")
        .api_keys(vec![
            "sk-live-abc123".to_string(),
            "sk-live-def456".to_string(),
        ]);

    let _server = ApiServerBuilder::new()
        .config(ApiConfig::default())
        .middleware(AuthMiddleware::new(auth_config))
        .build()?;

    Ok(())
}

/// Example: Custom error handling
#[allow(dead_code)]
fn custom_error_handler(req: Request) -> Response {
    match req.path() {
        "/not-found" => Response::error(Status::NotFound, "Resource not found"),
        "/forbidden" => Response::error(Status::Forbidden, "Access denied"),
        _ => Response::error(Status::InternalServerError, "Unknown error"),
    }
}
