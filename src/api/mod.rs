//! # API Module
//!
//! Real-time communication layer for the Panpsychism orchestration framework.
//!
//! ## Components
//!
//! - **WebSocket Server**: Bidirectional streaming for real-time query responses
//! - **REST API Server**: HTTP endpoints for query processing and agent management
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::api::{WebSocketServer, WebSocketConfig, ApiServer, ApiConfig};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     // WebSocket server
//!     let ws_config = WebSocketConfig::default();
//!     let ws_server = WebSocketServer::new(ws_config);
//!
//!     // REST API server
//!     let api_config = ApiConfig::default();
//!     let api_server = ApiServer::new(api_config);
//!
//!     // Handle connections...
//!     Ok(())
//! }
//! ```

pub mod websocket;
pub mod rest;

// Re-exports for convenient access - WebSocket
pub use websocket::{
    // Core types
    WebSocketServer,
    WebSocketServerBuilder,
    WebSocketConfig,
    Connection,
    ConnectionId,
    ConnectionState,
    ClientInfo,
    // Message types
    ClientMessage,
    ServerMessage,
    QueryResult,
    QueryRequest,
    // Handler types
    MessageHandler,
    QueryHandler,
    // Stats
    WebSocketStats,
};

// Re-exports for convenient access - REST API
pub use rest::{
    // Core types
    ApiServer,
    ApiServerBuilder,
    ApiConfig,
    Router,
    Route,
    // HTTP types
    Method,
    Status,
    Request,
    Response,
    ApiError,
    // Handler types
    RouteHandler,
    // Middleware
    Middleware,
    CorsMiddleware,
    RateLimitMiddleware,
    LoggingMiddleware,
    AuthMiddleware,
    // State and config
    ApiState,
    AgentInfo as RestAgentInfo,
    AgentStatus as RestAgentStatus,
    ServerMetrics,
    RateLimitConfig as RestRateLimitConfig,
    AuthConfig,
};
