//! # WebSocket Handler
//!
//! Real-time bidirectional communication for streaming responses in the
//! Panpsychism orchestration framework.
//!
//! ## Features
//!
//! - **Streaming Responses**: Token-by-token query response streaming
//! - **Topic Subscriptions**: Pub/sub pattern for event notifications
//! - **Connection Management**: Lifecycle tracking with heartbeat monitoring
//! - **Broadcast**: Send messages to all connections or groups
//! - **Graceful Shutdown**: Clean connection termination
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::api::{WebSocketServer, WebSocketConfig, ClientMessage, ServerMessage};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let config = WebSocketConfig::default();
//!     let server = WebSocketServerBuilder::new()
//!         .config(config)
//!         .build();
//!
//!     // Register query handler
//!     server.register_query_handler(|query, tx| {
//!         Box::pin(async move {
//!             // Stream response tokens
//!             tx.send(ServerMessage::QueryChunk {
//!                 id: query.id.clone(),
//!                 chunk: "Hello".to_string(),
//!                 done: false,
//!             }).await?;
//!             tx.send(ServerMessage::QueryChunk {
//!                 id: query.id,
//!                 chunk: " World!".to_string(),
//!                 done: true,
//!             }).await?;
//!             Ok(())
//!         })
//!     }).await;
//!
//!     Ok(())
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, mpsc, RwLock};
use uuid::Uuid;

use crate::Result;

// ============================================================================
// Type Aliases
// ============================================================================

/// Unique identifier for a WebSocket connection.
pub type ConnectionId = String;

/// Type alias for message handler functions.
pub type MessageHandler = Arc<
    dyn Fn(ClientMessage, Arc<Connection>) -> Pin<Box<dyn Future<Output = Result<Option<ServerMessage>>> + Send>>
        + Send
        + Sync,
>;

/// Type alias for query handler functions that support streaming.
pub type QueryHandler = Arc<
    dyn Fn(QueryRequest, mpsc::Sender<ServerMessage>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>
        + Send
        + Sync,
>;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the WebSocket server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    /// Maximum number of concurrent connections allowed.
    pub max_connections: usize,

    /// Interval for sending heartbeat pings (in seconds).
    pub heartbeat_interval_secs: u64,

    /// Timeout for considering a connection dead (in seconds).
    pub connection_timeout_secs: u64,

    /// Maximum size of a single message in bytes.
    pub max_message_size: usize,

    /// Interval for sending ping frames (in seconds).
    pub ping_interval_secs: u64,

    /// Buffer size for the broadcast channel.
    pub broadcast_buffer_size: usize,

    /// Buffer size for connection message queues.
    pub connection_buffer_size: usize,

    /// Whether to enable message compression.
    pub enable_compression: bool,

    /// Maximum number of subscriptions per connection.
    pub max_subscriptions_per_connection: usize,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            max_connections: 1000,
            heartbeat_interval_secs: 30,
            connection_timeout_secs: 90,
            max_message_size: 1024 * 1024, // 1MB
            ping_interval_secs: 15,
            broadcast_buffer_size: 1000,
            connection_buffer_size: 100,
            enable_compression: true,
            max_subscriptions_per_connection: 50,
        }
    }
}

impl WebSocketConfig {
    /// Create a new configuration with custom values.
    pub fn new(max_connections: usize, heartbeat_interval_secs: u64) -> Self {
        Self {
            max_connections,
            heartbeat_interval_secs,
            ..Default::default()
        }
    }

    /// Create a minimal configuration for testing.
    pub fn minimal() -> Self {
        Self {
            max_connections: 10,
            heartbeat_interval_secs: 5,
            connection_timeout_secs: 15,
            max_message_size: 10240,
            ping_interval_secs: 3,
            broadcast_buffer_size: 10,
            connection_buffer_size: 10,
            enable_compression: false,
            max_subscriptions_per_connection: 5,
        }
    }

    /// Create a high-performance configuration.
    pub fn high_performance() -> Self {
        Self {
            max_connections: 10000,
            heartbeat_interval_secs: 60,
            connection_timeout_secs: 180,
            max_message_size: 10 * 1024 * 1024, // 10MB
            ping_interval_secs: 30,
            broadcast_buffer_size: 10000,
            connection_buffer_size: 1000,
            enable_compression: true,
            max_subscriptions_per_connection: 100,
        }
    }
}

// ============================================================================
// Connection State
// ============================================================================

/// State of a WebSocket connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConnectionState {
    /// Connection is being established.
    Connecting,
    /// Connection is active and ready for messages.
    Connected,
    /// Connection is in the process of closing.
    Closing,
    /// Connection has been closed.
    Closed,
}

impl Default for ConnectionState {
    fn default() -> Self {
        ConnectionState::Connecting
    }
}

impl fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConnectionState::Connecting => write!(f, "Connecting"),
            ConnectionState::Connected => write!(f, "Connected"),
            ConnectionState::Closing => write!(f, "Closing"),
            ConnectionState::Closed => write!(f, "Closed"),
        }
    }
}

// ============================================================================
// Client Info
// ============================================================================

/// Information about a connected client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client's IP address (if available).
    pub ip_address: Option<String>,

    /// Client's user agent string.
    pub user_agent: Option<String>,

    /// Authentication token or user ID (if authenticated).
    pub auth_token: Option<String>,

    /// User ID associated with this connection.
    pub user_id: Option<String>,

    /// Client-provided metadata.
    pub metadata: HashMap<String, String>,
}

impl Default for ClientInfo {
    fn default() -> Self {
        Self {
            ip_address: None,
            user_agent: None,
            auth_token: None,
            user_id: None,
            metadata: HashMap::new(),
        }
    }
}

impl ClientInfo {
    /// Create a new client info with IP address.
    pub fn with_ip(ip: impl Into<String>) -> Self {
        Self {
            ip_address: Some(ip.into()),
            ..Default::default()
        }
    }

    /// Set the user agent.
    pub fn user_agent(mut self, agent: impl Into<String>) -> Self {
        self.user_agent = Some(agent.into());
        self
    }

    /// Set the auth token.
    pub fn auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Set the user ID.
    pub fn user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if the client is authenticated.
    pub fn is_authenticated(&self) -> bool {
        self.auth_token.is_some() || self.user_id.is_some()
    }
}

// ============================================================================
// Connection
// ============================================================================

/// Represents a WebSocket connection.
#[derive(Debug)]
pub struct Connection {
    /// Unique connection identifier.
    pub id: ConnectionId,

    /// Information about the connected client.
    pub client_info: ClientInfo,

    /// Topics this connection is subscribed to.
    pub subscriptions: RwLock<HashSet<String>>,

    /// Timestamp when the connection was created.
    pub created_at: Instant,

    /// Timestamp of the last activity on this connection.
    pub last_activity: RwLock<Instant>,

    /// Current state of the connection.
    pub state: RwLock<ConnectionState>,

    /// Groups this connection belongs to.
    pub groups: RwLock<HashSet<String>>,

    /// Connection-specific metadata.
    pub metadata: RwLock<HashMap<String, serde_json::Value>>,
}

impl Connection {
    /// Create a new connection.
    pub fn new(client_info: ClientInfo) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4().to_string(),
            client_info,
            subscriptions: RwLock::new(HashSet::new()),
            created_at: now,
            last_activity: RwLock::new(now),
            state: RwLock::new(ConnectionState::Connecting),
            groups: RwLock::new(HashSet::new()),
            metadata: RwLock::new(HashMap::new()),
        }
    }

    /// Create a connection with a specific ID.
    pub fn with_id(id: impl Into<ConnectionId>, client_info: ClientInfo) -> Self {
        let now = Instant::now();
        Self {
            id: id.into(),
            client_info,
            subscriptions: RwLock::new(HashSet::new()),
            created_at: now,
            last_activity: RwLock::new(now),
            state: RwLock::new(ConnectionState::Connecting),
            groups: RwLock::new(HashSet::new()),
            metadata: RwLock::new(HashMap::new()),
        }
    }

    /// Update the last activity timestamp.
    pub async fn touch(&self) {
        *self.last_activity.write().await = Instant::now();
    }

    /// Get the connection age in seconds.
    pub fn age_secs(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }

    /// Check if the connection is stale based on timeout.
    pub async fn is_stale(&self, timeout_secs: u64) -> bool {
        self.last_activity.read().await.elapsed().as_secs() > timeout_secs
    }

    /// Get the current state.
    pub async fn get_state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Set the connection state.
    pub async fn set_state(&self, state: ConnectionState) {
        *self.state.write().await = state;
    }

    /// Subscribe to a topic.
    pub async fn subscribe(&self, topic: impl Into<String>) -> bool {
        self.subscriptions.write().await.insert(topic.into())
    }

    /// Unsubscribe from a topic.
    pub async fn unsubscribe(&self, topic: &str) -> bool {
        self.subscriptions.write().await.remove(topic)
    }

    /// Check if subscribed to a topic.
    pub async fn is_subscribed(&self, topic: &str) -> bool {
        self.subscriptions.read().await.contains(topic)
    }

    /// Get all subscriptions.
    pub async fn get_subscriptions(&self) -> HashSet<String> {
        self.subscriptions.read().await.clone()
    }

    /// Get subscription count.
    pub async fn subscription_count(&self) -> usize {
        self.subscriptions.read().await.len()
    }

    /// Join a group.
    pub async fn join_group(&self, group: impl Into<String>) -> bool {
        self.groups.write().await.insert(group.into())
    }

    /// Leave a group.
    pub async fn leave_group(&self, group: &str) -> bool {
        self.groups.write().await.remove(group)
    }

    /// Check if in a group.
    pub async fn is_in_group(&self, group: &str) -> bool {
        self.groups.read().await.contains(group)
    }

    /// Set metadata value.
    pub async fn set_metadata(&self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.write().await.insert(key.into(), value);
    }

    /// Get metadata value.
    pub async fn get_metadata(&self, key: &str) -> Option<serde_json::Value> {
        self.metadata.read().await.get(key).cloned()
    }
}

// ============================================================================
// Client Messages (from client to server)
// ============================================================================

/// Messages sent from client to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Submit a query for processing.
    Query {
        /// Unique query identifier.
        id: String,
        /// Query text.
        text: String,
        /// Optional agent to route the query to.
        agent: Option<String>,
        /// Optional additional parameters.
        #[serde(default)]
        params: HashMap<String, serde_json::Value>,
    },

    /// Subscribe to topics.
    Subscribe {
        /// Topics to subscribe to.
        topics: Vec<String>,
    },

    /// Unsubscribe from topics.
    Unsubscribe {
        /// Topics to unsubscribe from.
        topics: Vec<String>,
    },

    /// Ping message for connection health.
    Ping,

    /// Cancel an ongoing query.
    CancelQuery {
        /// ID of the query to cancel.
        id: String,
    },

    /// Join a group/room.
    JoinGroup {
        /// Group name.
        group: String,
    },

    /// Leave a group/room.
    LeaveGroup {
        /// Group name.
        group: String,
    },

    /// Authentication message.
    Authenticate {
        /// Authentication token.
        token: String,
    },
}

impl ClientMessage {
    /// Create a new query message.
    pub fn query(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Query {
            id: id.into(),
            text: text.into(),
            agent: None,
            params: HashMap::new(),
        }
    }

    /// Create a new query message with agent.
    pub fn query_with_agent(
        id: impl Into<String>,
        text: impl Into<String>,
        agent: impl Into<String>,
    ) -> Self {
        Self::Query {
            id: id.into(),
            text: text.into(),
            agent: Some(agent.into()),
            params: HashMap::new(),
        }
    }

    /// Create a subscribe message.
    pub fn subscribe(topics: Vec<String>) -> Self {
        Self::Subscribe { topics }
    }

    /// Create an unsubscribe message.
    pub fn unsubscribe(topics: Vec<String>) -> Self {
        Self::Unsubscribe { topics }
    }

    /// Create a ping message.
    pub fn ping() -> Self {
        Self::Ping
    }

    /// Create a cancel query message.
    pub fn cancel_query(id: impl Into<String>) -> Self {
        Self::CancelQuery { id: id.into() }
    }

    /// Get the message type as a string.
    pub fn message_type(&self) -> &'static str {
        match self {
            Self::Query { .. } => "query",
            Self::Subscribe { .. } => "subscribe",
            Self::Unsubscribe { .. } => "unsubscribe",
            Self::Ping => "ping",
            Self::CancelQuery { .. } => "cancel_query",
            Self::JoinGroup { .. } => "join_group",
            Self::LeaveGroup { .. } => "leave_group",
            Self::Authenticate { .. } => "authenticate",
        }
    }
}

// ============================================================================
// Server Messages (from server to client)
// ============================================================================

/// Query request structure for handlers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Unique query identifier.
    pub id: String,
    /// Query text.
    pub text: String,
    /// Optional agent to route to.
    pub agent: Option<String>,
    /// Additional parameters.
    pub params: HashMap<String, serde_json::Value>,
    /// Connection ID that sent the query.
    pub connection_id: ConnectionId,
}

/// Result of a completed query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Full response text.
    pub response: String,
    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
    /// Number of tokens in the response.
    pub token_count: usize,
    /// Agent that processed the query.
    pub agent: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl QueryResult {
    /// Create a new query result.
    pub fn new(response: impl Into<String>, processing_time_ms: u64) -> Self {
        Self {
            response: response.into(),
            processing_time_ms,
            token_count: 0,
            agent: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the token count.
    pub fn with_token_count(mut self, count: usize) -> Self {
        self.token_count = count;
        self
    }

    /// Set the agent.
    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agent = Some(agent.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Messages sent from server to client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Query processing has started.
    QueryStart {
        /// Query identifier.
        id: String,
    },

    /// A chunk of the query response (streaming).
    QueryChunk {
        /// Query identifier.
        id: String,
        /// Response chunk.
        chunk: String,
        /// Whether this is the final chunk.
        done: bool,
    },

    /// Query completed successfully.
    QueryComplete {
        /// Query identifier.
        id: String,
        /// Full result.
        result: QueryResult,
    },

    /// Query failed with an error.
    QueryError {
        /// Query identifier.
        id: String,
        /// Error message.
        error: String,
        /// Error code (if applicable).
        code: Option<String>,
    },

    /// An event on a subscribed topic.
    Event {
        /// Topic the event is for.
        topic: String,
        /// Event data.
        data: serde_json::Value,
        /// Timestamp.
        timestamp: DateTime<Utc>,
    },

    /// Response to a ping.
    Pong,

    /// Subscription confirmation.
    Subscribed {
        /// Topics successfully subscribed to.
        topics: Vec<String>,
    },

    /// Unsubscription confirmation.
    Unsubscribed {
        /// Topics unsubscribed from.
        topics: Vec<String>,
    },

    /// Query was cancelled.
    QueryCancelled {
        /// Query identifier.
        id: String,
    },

    /// Server error.
    Error {
        /// Error message.
        message: String,
        /// Error code.
        code: Option<String>,
    },

    /// Authentication result.
    Authenticated {
        /// Whether authentication succeeded.
        success: bool,
        /// User ID if authenticated.
        user_id: Option<String>,
        /// Error message if failed.
        error: Option<String>,
    },

    /// Connection established confirmation.
    Connected {
        /// Connection ID assigned.
        connection_id: ConnectionId,
        /// Server version.
        server_version: String,
    },

    /// Reconnection hint.
    ReconnectHint {
        /// Reason for reconnection suggestion.
        reason: String,
        /// Delay before reconnecting (seconds).
        delay_secs: u64,
    },
}

impl ServerMessage {
    /// Create a query start message.
    pub fn query_start(id: impl Into<String>) -> Self {
        Self::QueryStart { id: id.into() }
    }

    /// Create a query chunk message.
    pub fn query_chunk(id: impl Into<String>, chunk: impl Into<String>, done: bool) -> Self {
        Self::QueryChunk {
            id: id.into(),
            chunk: chunk.into(),
            done,
        }
    }

    /// Create a query complete message.
    pub fn query_complete(id: impl Into<String>, result: QueryResult) -> Self {
        Self::QueryComplete {
            id: id.into(),
            result,
        }
    }

    /// Create a query error message.
    pub fn query_error(id: impl Into<String>, error: impl Into<String>) -> Self {
        Self::QueryError {
            id: id.into(),
            error: error.into(),
            code: None,
        }
    }

    /// Create a query error message with code.
    pub fn query_error_with_code(
        id: impl Into<String>,
        error: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self::QueryError {
            id: id.into(),
            error: error.into(),
            code: Some(code.into()),
        }
    }

    /// Create an event message.
    pub fn event(topic: impl Into<String>, data: serde_json::Value) -> Self {
        Self::Event {
            topic: topic.into(),
            data,
            timestamp: Utc::now(),
        }
    }

    /// Create a pong message.
    pub fn pong() -> Self {
        Self::Pong
    }

    /// Create a subscribed message.
    pub fn subscribed(topics: Vec<String>) -> Self {
        Self::Subscribed { topics }
    }

    /// Create an unsubscribed message.
    pub fn unsubscribed(topics: Vec<String>) -> Self {
        Self::Unsubscribed { topics }
    }

    /// Create a query cancelled message.
    pub fn query_cancelled(id: impl Into<String>) -> Self {
        Self::QueryCancelled { id: id.into() }
    }

    /// Create an error message.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
            code: None,
        }
    }

    /// Create an error message with code.
    pub fn error_with_code(message: impl Into<String>, code: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
            code: Some(code.into()),
        }
    }

    /// Create an authenticated message.
    pub fn authenticated(success: bool, user_id: Option<String>) -> Self {
        Self::Authenticated {
            success,
            user_id,
            error: None,
        }
    }

    /// Create a connected message.
    pub fn connected(connection_id: ConnectionId, server_version: impl Into<String>) -> Self {
        Self::Connected {
            connection_id,
            server_version: server_version.into(),
        }
    }

    /// Create a reconnect hint message.
    pub fn reconnect_hint(reason: impl Into<String>, delay_secs: u64) -> Self {
        Self::ReconnectHint {
            reason: reason.into(),
            delay_secs,
        }
    }

    /// Get the message type as a string.
    pub fn message_type(&self) -> &'static str {
        match self {
            Self::QueryStart { .. } => "query_start",
            Self::QueryChunk { .. } => "query_chunk",
            Self::QueryComplete { .. } => "query_complete",
            Self::QueryError { .. } => "query_error",
            Self::Event { .. } => "event",
            Self::Pong => "pong",
            Self::Subscribed { .. } => "subscribed",
            Self::Unsubscribed { .. } => "unsubscribed",
            Self::QueryCancelled { .. } => "query_cancelled",
            Self::Error { .. } => "error",
            Self::Authenticated { .. } => "authenticated",
            Self::Connected { .. } => "connected",
            Self::ReconnectHint { .. } => "reconnect_hint",
        }
    }

    /// Check if this is an error message.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::QueryError { .. } | Self::Error { .. })
    }
}

// ============================================================================
// WebSocket Statistics
// ============================================================================

/// Statistics about the WebSocket server.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct WebSocketStats {
    /// Total connections ever made.
    pub total_connections: u64,
    /// Currently active connections.
    pub active_connections: u64,
    /// Total messages received.
    pub messages_received: u64,
    /// Total messages sent.
    pub messages_sent: u64,
    /// Total queries processed.
    pub queries_processed: u64,
    /// Total query errors.
    pub query_errors: u64,
    /// Total broadcasts sent.
    pub broadcasts_sent: u64,
    /// Total events published.
    pub events_published: u64,
    /// Total pings received.
    pub pings_received: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
}

impl WebSocketStats {
    /// Create a new stats instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new connection.
    pub fn record_connection(&mut self) {
        self.total_connections += 1;
        self.active_connections += 1;
    }

    /// Record a disconnection.
    pub fn record_disconnection(&mut self) {
        if self.active_connections > 0 {
            self.active_connections -= 1;
        }
    }

    /// Record a received message.
    pub fn record_message_received(&mut self, bytes: usize) {
        self.messages_received += 1;
        self.bytes_received += bytes as u64;
    }

    /// Record a sent message.
    pub fn record_message_sent(&mut self, bytes: usize) {
        self.messages_sent += 1;
        self.bytes_sent += bytes as u64;
    }

    /// Record a processed query.
    pub fn record_query(&mut self) {
        self.queries_processed += 1;
    }

    /// Record a query error.
    pub fn record_query_error(&mut self) {
        self.query_errors += 1;
    }

    /// Record a broadcast.
    pub fn record_broadcast(&mut self) {
        self.broadcasts_sent += 1;
    }

    /// Record an event.
    pub fn record_event(&mut self) {
        self.events_published += 1;
    }

    /// Record a ping.
    pub fn record_ping(&mut self) {
        self.pings_received += 1;
    }
}

// ============================================================================
// WebSocket Server
// ============================================================================

/// The WebSocket server for handling real-time connections.
pub struct WebSocketServer {
    /// Server configuration.
    config: WebSocketConfig,
    /// Active connections.
    connections: Arc<RwLock<HashMap<ConnectionId, Arc<Connection>>>>,
    /// Topic subscriptions: topic -> set of connection IDs.
    topic_subscriptions: Arc<RwLock<HashMap<String, HashSet<ConnectionId>>>>,
    /// Group memberships: group -> set of connection IDs.
    groups: Arc<RwLock<HashMap<String, HashSet<ConnectionId>>>>,
    /// Broadcast channel sender.
    broadcast_tx: broadcast::Sender<ServerMessage>,
    /// Query handler.
    query_handler: Arc<RwLock<Option<QueryHandler>>>,
    /// Server statistics.
    stats: Arc<RwLock<WebSocketStats>>,
    /// Shutdown flag.
    shutdown: Arc<RwLock<bool>>,
    /// Pending queries (for cancellation).
    pending_queries: Arc<RwLock<HashMap<String, tokio::sync::oneshot::Sender<()>>>>,
}

impl fmt::Debug for WebSocketServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebSocketServer")
            .field("config", &self.config)
            .field("connections", &"<connections>")
            .field("topic_subscriptions", &"<topic_subscriptions>")
            .field("groups", &"<groups>")
            .field("query_handler", &"<query_handler>")
            .field("stats", &"<stats>")
            .field("shutdown", &"<shutdown>")
            .finish()
    }
}

impl WebSocketServer {
    /// Create a new WebSocket server with the given configuration.
    pub fn new(config: WebSocketConfig) -> Self {
        let (broadcast_tx, _) = broadcast::channel(config.broadcast_buffer_size);

        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            topic_subscriptions: Arc::new(RwLock::new(HashMap::new())),
            groups: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
            query_handler: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(WebSocketStats::new())),
            shutdown: Arc::new(RwLock::new(false)),
            pending_queries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new server with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(WebSocketConfig::default())
    }

    /// Get the server configuration.
    pub fn config(&self) -> &WebSocketConfig {
        &self.config
    }

    /// Get a snapshot of the current statistics.
    pub async fn stats(&self) -> WebSocketStats {
        self.stats.read().await.clone()
    }

    /// Get the number of active connections.
    pub async fn connection_count(&self) -> usize {
        self.connections.read().await.len()
    }

    /// Check if the server is at capacity.
    pub async fn is_at_capacity(&self) -> bool {
        self.connections.read().await.len() >= self.config.max_connections
    }

    /// Check if the server is shutting down.
    pub async fn is_shutdown(&self) -> bool {
        *self.shutdown.read().await
    }

    // ========================================================================
    // Query Handler
    // ========================================================================

    /// Register a query handler for processing queries.
    pub async fn register_query_handler<F>(&self, handler: F)
    where
        F: Fn(QueryRequest, mpsc::Sender<ServerMessage>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        *self.query_handler.write().await = Some(Arc::new(handler));
    }

    /// Check if a query handler is registered.
    pub async fn has_query_handler(&self) -> bool {
        self.query_handler.read().await.is_some()
    }

    // ========================================================================
    // Connection Management
    // ========================================================================

    /// Accept a new connection.
    pub async fn accept_connection(&self, client_info: ClientInfo) -> Result<Arc<Connection>> {
        // Check capacity
        if self.is_at_capacity().await {
            return Err(crate::Error::Internal {
                message: "Server at maximum connection capacity".to_string(),
            });
        }

        // Check shutdown
        if self.is_shutdown().await {
            return Err(crate::Error::Internal {
                message: "Server is shutting down".to_string(),
            });
        }

        let connection = Arc::new(Connection::new(client_info));
        connection.set_state(ConnectionState::Connected).await;

        // Add to connections map
        self.connections
            .write()
            .await
            .insert(connection.id.clone(), connection.clone());

        // Update stats
        self.stats.write().await.record_connection();

        Ok(connection)
    }

    /// Accept a connection with a specific ID.
    pub async fn accept_connection_with_id(
        &self,
        id: impl Into<ConnectionId>,
        client_info: ClientInfo,
    ) -> Result<Arc<Connection>> {
        let id = id.into();

        // Check if ID already exists
        if self.connections.read().await.contains_key(&id) {
            return Err(crate::Error::Internal {
                message: format!("Connection ID already exists: {}", id),
            });
        }

        // Check capacity
        if self.is_at_capacity().await {
            return Err(crate::Error::Internal {
                message: "Server at maximum connection capacity".to_string(),
            });
        }

        let connection = Arc::new(Connection::with_id(id.clone(), client_info));
        connection.set_state(ConnectionState::Connected).await;

        self.connections
            .write()
            .await
            .insert(id, connection.clone());

        self.stats.write().await.record_connection();

        Ok(connection)
    }

    /// Get a connection by ID.
    pub async fn get_connection(&self, id: &str) -> Option<Arc<Connection>> {
        self.connections.read().await.get(id).cloned()
    }

    /// Get all active connection IDs.
    pub async fn get_connection_ids(&self) -> Vec<ConnectionId> {
        self.connections.read().await.keys().cloned().collect()
    }

    /// Close a connection.
    pub async fn close_connection(&self, id: &str) -> bool {
        let connection = match self.connections.write().await.remove(id) {
            Some(c) => c,
            None => return false,
        };

        // Update state
        connection.set_state(ConnectionState::Closed).await;

        // Remove from all topic subscriptions
        let subs = connection.get_subscriptions().await;
        let mut topic_subs = self.topic_subscriptions.write().await;
        for topic in subs {
            if let Some(subscribers) = topic_subs.get_mut(&topic) {
                subscribers.remove(id);
                if subscribers.is_empty() {
                    topic_subs.remove(&topic);
                }
            }
        }

        // Remove from all groups
        let groups = connection.groups.read().await.clone();
        let mut group_map = self.groups.write().await;
        for group in groups {
            if let Some(members) = group_map.get_mut(&group) {
                members.remove(id);
                if members.is_empty() {
                    group_map.remove(&group);
                }
            }
        }

        // Update stats
        self.stats.write().await.record_disconnection();

        true
    }

    /// Close stale connections based on timeout.
    pub async fn close_stale_connections(&self) -> Vec<ConnectionId> {
        let timeout = self.config.connection_timeout_secs;
        let mut stale = Vec::new();

        // Find stale connections
        let connections = self.connections.read().await;
        for (id, conn) in connections.iter() {
            if conn.is_stale(timeout).await {
                stale.push(id.clone());
            }
        }
        drop(connections);

        // Close them
        for id in &stale {
            self.close_connection(id).await;
        }

        stale
    }

    // ========================================================================
    // Subscription Management
    // ========================================================================

    /// Subscribe a connection to topics.
    pub async fn subscribe(
        &self,
        connection_id: &str,
        topics: Vec<String>,
    ) -> Result<Vec<String>> {
        let connection = self.get_connection(connection_id).await.ok_or_else(|| {
            crate::Error::Internal {
                message: format!("Connection not found: {}", connection_id),
            }
        })?;

        let max_subs = self.config.max_subscriptions_per_connection;
        let current_count = connection.subscription_count().await;

        let mut subscribed = Vec::new();
        let mut topic_subs = self.topic_subscriptions.write().await;

        for topic in topics {
            if current_count + subscribed.len() >= max_subs {
                break;
            }

            if connection.subscribe(&topic).await {
                topic_subs
                    .entry(topic.clone())
                    .or_insert_with(HashSet::new)
                    .insert(connection_id.to_string());
                subscribed.push(topic);
            }
        }

        Ok(subscribed)
    }

    /// Unsubscribe a connection from topics.
    pub async fn unsubscribe(
        &self,
        connection_id: &str,
        topics: Vec<String>,
    ) -> Result<Vec<String>> {
        let connection = self.get_connection(connection_id).await.ok_or_else(|| {
            crate::Error::Internal {
                message: format!("Connection not found: {}", connection_id),
            }
        })?;

        let mut unsubscribed = Vec::new();
        let mut topic_subs = self.topic_subscriptions.write().await;

        for topic in topics {
            if connection.unsubscribe(&topic).await {
                if let Some(subscribers) = topic_subs.get_mut(&topic) {
                    subscribers.remove(connection_id);
                    if subscribers.is_empty() {
                        topic_subs.remove(&topic);
                    }
                }
                unsubscribed.push(topic);
            }
        }

        Ok(unsubscribed)
    }

    /// Get all subscribers for a topic.
    pub async fn get_topic_subscribers(&self, topic: &str) -> Vec<ConnectionId> {
        self.topic_subscriptions
            .read()
            .await
            .get(topic)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get all active topics.
    pub async fn get_topics(&self) -> Vec<String> {
        self.topic_subscriptions
            .read()
            .await
            .keys()
            .cloned()
            .collect()
    }

    // ========================================================================
    // Group Management
    // ========================================================================

    /// Add a connection to a group.
    pub async fn join_group(&self, connection_id: &str, group: &str) -> Result<bool> {
        let connection = self.get_connection(connection_id).await.ok_or_else(|| {
            crate::Error::Internal {
                message: format!("Connection not found: {}", connection_id),
            }
        })?;

        if connection.join_group(group).await {
            self.groups
                .write()
                .await
                .entry(group.to_string())
                .or_insert_with(HashSet::new)
                .insert(connection_id.to_string());
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Remove a connection from a group.
    pub async fn leave_group(&self, connection_id: &str, group: &str) -> Result<bool> {
        let connection = self.get_connection(connection_id).await.ok_or_else(|| {
            crate::Error::Internal {
                message: format!("Connection not found: {}", connection_id),
            }
        })?;

        if connection.leave_group(group).await {
            let mut groups = self.groups.write().await;
            if let Some(members) = groups.get_mut(group) {
                members.remove(connection_id);
                if members.is_empty() {
                    groups.remove(group);
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get all members of a group.
    pub async fn get_group_members(&self, group: &str) -> Vec<ConnectionId> {
        self.groups
            .read()
            .await
            .get(group)
            .map(|m| m.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get all groups.
    pub async fn get_groups(&self) -> Vec<String> {
        self.groups.read().await.keys().cloned().collect()
    }

    // ========================================================================
    // Message Handling
    // ========================================================================

    /// Handle an incoming client message.
    pub async fn handle_message(
        &self,
        connection_id: &str,
        message: ClientMessage,
    ) -> Result<Option<ServerMessage>> {
        // Update connection activity
        if let Some(connection) = self.get_connection(connection_id).await {
            connection.touch().await;
        }

        match message {
            ClientMessage::Ping => {
                self.stats.write().await.record_ping();
                Ok(Some(ServerMessage::pong()))
            }

            ClientMessage::Subscribe { topics } => {
                let subscribed = self.subscribe(connection_id, topics).await?;
                Ok(Some(ServerMessage::subscribed(subscribed)))
            }

            ClientMessage::Unsubscribe { topics } => {
                let unsubscribed = self.unsubscribe(connection_id, topics).await?;
                Ok(Some(ServerMessage::unsubscribed(unsubscribed)))
            }

            ClientMessage::Query { id, text, agent, params } => {
                self.handle_query(connection_id, id, text, agent, params).await
            }

            ClientMessage::CancelQuery { id } => {
                self.cancel_query(&id).await;
                Ok(Some(ServerMessage::query_cancelled(id)))
            }

            ClientMessage::JoinGroup { group } => {
                self.join_group(connection_id, &group).await?;
                Ok(None)
            }

            ClientMessage::LeaveGroup { group } => {
                self.leave_group(connection_id, &group).await?;
                Ok(None)
            }

            ClientMessage::Authenticate { token } => {
                // Basic auth handling - in production, implement proper validation
                if let Some(connection) = self.get_connection(connection_id).await {
                    connection.set_metadata("auth_token", serde_json::json!(token)).await;
                    Ok(Some(ServerMessage::authenticated(true, None)))
                } else {
                    Ok(Some(ServerMessage::authenticated(false, None)))
                }
            }
        }
    }

    /// Handle a query request.
    async fn handle_query(
        &self,
        connection_id: &str,
        id: String,
        text: String,
        agent: Option<String>,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<Option<ServerMessage>> {
        let handler = self.query_handler.read().await.clone();

        match handler {
            Some(handler) => {
                let request = QueryRequest {
                    id: id.clone(),
                    text,
                    agent,
                    params,
                    connection_id: connection_id.to_string(),
                };

                // Create cancellation channel
                let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
                self.pending_queries.write().await.insert(id.clone(), cancel_tx);

                // Create response channel
                let (response_tx, mut response_rx) = mpsc::channel(100);

                // Spawn handler task
                let handler_clone = handler.clone();
                let request_clone = request.clone();
                let query_id = id.clone();
                let stats = self.stats.clone();

                tokio::spawn(async move {
                    tokio::select! {
                        result = handler_clone(request_clone, response_tx.clone()) => {
                            if let Err(e) = result {
                                let _ = response_tx.send(ServerMessage::query_error(&query_id, e.to_string())).await;
                                stats.write().await.record_query_error();
                            } else {
                                stats.write().await.record_query();
                            }
                        }
                        _ = &mut cancel_rx => {
                            let _ = response_tx.send(ServerMessage::query_cancelled(&query_id)).await;
                        }
                    }
                });

                // Return the start message and let responses flow through the channel
                // In a real implementation, you'd pipe response_rx to the WebSocket
                drop(response_rx); // For now, we just acknowledge

                Ok(Some(ServerMessage::query_start(id)))
            }
            None => {
                Ok(Some(ServerMessage::query_error(id, "No query handler registered")))
            }
        }
    }

    /// Cancel a pending query.
    pub async fn cancel_query(&self, query_id: &str) -> bool {
        if let Some(cancel_tx) = self.pending_queries.write().await.remove(query_id) {
            let _ = cancel_tx.send(());
            true
        } else {
            false
        }
    }

    // ========================================================================
    // Broadcasting
    // ========================================================================

    /// Broadcast a message to all connections.
    pub async fn broadcast(&self, message: ServerMessage) -> usize {
        self.stats.write().await.record_broadcast();
        let _ = self.broadcast_tx.send(message);
        self.connection_count().await
    }

    /// Send a message to subscribers of a topic.
    pub async fn publish(&self, topic: &str, data: serde_json::Value) -> usize {
        let subscribers = self.get_topic_subscribers(topic).await;
        let count = subscribers.len();

        if count > 0 {
            self.stats.write().await.record_event();
            let message = ServerMessage::event(topic, data);
            let _ = self.broadcast_tx.send(message);
        }

        count
    }

    /// Send a message to members of a group.
    pub async fn send_to_group(&self, group: &str, message: ServerMessage) -> usize {
        let members = self.get_group_members(group).await;
        let count = members.len();

        if count > 0 {
            let _ = self.broadcast_tx.send(message);
        }

        count
    }

    /// Get a receiver for broadcast messages.
    pub fn subscribe_broadcast(&self) -> broadcast::Receiver<ServerMessage> {
        self.broadcast_tx.subscribe()
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /// Initiate graceful shutdown.
    pub async fn shutdown(&self) {
        *self.shutdown.write().await = true;

        // Cancel all pending queries
        let pending: Vec<_> = self.pending_queries.write().await.drain().collect();
        for (_, cancel_tx) in pending {
            let _ = cancel_tx.send(());
        }

        // Send reconnect hints to all connections
        let reconnect_msg = ServerMessage::reconnect_hint("Server shutting down", 5);
        let _ = self.broadcast_tx.send(reconnect_msg);

        // Close all connections
        let connection_ids: Vec<_> = self.get_connection_ids().await;
        for id in connection_ids {
            self.close_connection(&id).await;
        }
    }

    /// Reset the server (for testing).
    pub async fn reset(&self) {
        *self.shutdown.write().await = false;
        self.connections.write().await.clear();
        self.topic_subscriptions.write().await.clear();
        self.groups.write().await.clear();
        self.pending_queries.write().await.clear();
        *self.stats.write().await = WebSocketStats::new();
    }
}

// ============================================================================
// WebSocket Server Builder
// ============================================================================

/// Builder for WebSocketServer.
#[derive(Debug, Default)]
pub struct WebSocketServerBuilder {
    config: Option<WebSocketConfig>,
}

impl WebSocketServerBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the configuration.
    pub fn config(mut self, config: WebSocketConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the maximum number of connections.
    pub fn max_connections(mut self, max: usize) -> Self {
        let config = self.config.get_or_insert_with(WebSocketConfig::default);
        config.max_connections = max;
        self
    }

    /// Set the heartbeat interval.
    pub fn heartbeat_interval_secs(mut self, secs: u64) -> Self {
        let config = self.config.get_or_insert_with(WebSocketConfig::default);
        config.heartbeat_interval_secs = secs;
        self
    }

    /// Set the connection timeout.
    pub fn connection_timeout_secs(mut self, secs: u64) -> Self {
        let config = self.config.get_or_insert_with(WebSocketConfig::default);
        config.connection_timeout_secs = secs;
        self
    }

    /// Set the maximum message size.
    pub fn max_message_size(mut self, size: usize) -> Self {
        let config = self.config.get_or_insert_with(WebSocketConfig::default);
        config.max_message_size = size;
        self
    }

    /// Set the ping interval.
    pub fn ping_interval_secs(mut self, secs: u64) -> Self {
        let config = self.config.get_or_insert_with(WebSocketConfig::default);
        config.ping_interval_secs = secs;
        self
    }

    /// Enable or disable compression.
    pub fn enable_compression(mut self, enable: bool) -> Self {
        let config = self.config.get_or_insert_with(WebSocketConfig::default);
        config.enable_compression = enable;
        self
    }

    /// Build the WebSocket server.
    pub fn build(self) -> WebSocketServer {
        WebSocketServer::new(self.config.unwrap_or_default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    // ------------------------------------------------------------------------
    // Configuration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = WebSocketConfig::default();
        assert_eq!(config.max_connections, 1000);
        assert_eq!(config.heartbeat_interval_secs, 30);
        assert_eq!(config.connection_timeout_secs, 90);
        assert_eq!(config.max_message_size, 1024 * 1024);
        assert!(config.enable_compression);
    }

    #[test]
    fn test_config_custom() {
        let config = WebSocketConfig::new(500, 60);
        assert_eq!(config.max_connections, 500);
        assert_eq!(config.heartbeat_interval_secs, 60);
    }

    #[test]
    fn test_config_minimal() {
        let config = WebSocketConfig::minimal();
        assert_eq!(config.max_connections, 10);
        assert!(!config.enable_compression);
    }

    #[test]
    fn test_config_high_performance() {
        let config = WebSocketConfig::high_performance();
        assert_eq!(config.max_connections, 10000);
        assert_eq!(config.max_message_size, 10 * 1024 * 1024);
    }

    // ------------------------------------------------------------------------
    // Connection State Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_connection_state_default() {
        let state: ConnectionState = Default::default();
        assert_eq!(state, ConnectionState::Connecting);
    }

    #[test]
    fn test_connection_state_display() {
        assert_eq!(format!("{}", ConnectionState::Connecting), "Connecting");
        assert_eq!(format!("{}", ConnectionState::Connected), "Connected");
        assert_eq!(format!("{}", ConnectionState::Closing), "Closing");
        assert_eq!(format!("{}", ConnectionState::Closed), "Closed");
    }

    // ------------------------------------------------------------------------
    // Client Info Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_client_info_default() {
        let info = ClientInfo::default();
        assert!(info.ip_address.is_none());
        assert!(info.user_agent.is_none());
        assert!(!info.is_authenticated());
    }

    #[test]
    fn test_client_info_builder() {
        let info = ClientInfo::with_ip("192.168.1.1")
            .user_agent("Mozilla/5.0")
            .auth_token("token123")
            .user_id("user456")
            .with_metadata("device", "mobile");

        assert_eq!(info.ip_address, Some("192.168.1.1".to_string()));
        assert_eq!(info.user_agent, Some("Mozilla/5.0".to_string()));
        assert!(info.is_authenticated());
        assert_eq!(info.metadata.get("device"), Some(&"mobile".to_string()));
    }

    // ------------------------------------------------------------------------
    // Connection Tests
    // ------------------------------------------------------------------------

    #[tokio::test]
    async fn test_connection_new() {
        let conn = Connection::new(ClientInfo::default());
        assert!(!conn.id.is_empty());
        assert_eq!(conn.get_state().await, ConnectionState::Connecting);
        assert!(conn.get_subscriptions().await.is_empty());
    }

    #[tokio::test]
    async fn test_connection_with_id() {
        let conn = Connection::with_id("conn-123", ClientInfo::default());
        assert_eq!(conn.id, "conn-123");
    }

    #[tokio::test]
    async fn test_connection_touch() {
        let conn = Connection::new(ClientInfo::default());
        tokio::time::sleep(Duration::from_millis(10)).await;
        let before = *conn.last_activity.read().await;
        conn.touch().await;
        let after = *conn.last_activity.read().await;
        assert!(after > before);
    }

    #[tokio::test]
    async fn test_connection_state() {
        let conn = Connection::new(ClientInfo::default());
        assert_eq!(conn.get_state().await, ConnectionState::Connecting);

        conn.set_state(ConnectionState::Connected).await;
        assert_eq!(conn.get_state().await, ConnectionState::Connected);

        conn.set_state(ConnectionState::Closed).await;
        assert_eq!(conn.get_state().await, ConnectionState::Closed);
    }

    #[tokio::test]
    async fn test_connection_subscriptions() {
        let conn = Connection::new(ClientInfo::default());

        assert!(conn.subscribe("topic1").await);
        assert!(conn.subscribe("topic2").await);
        assert!(!conn.subscribe("topic1").await); // Already subscribed

        assert!(conn.is_subscribed("topic1").await);
        assert!(conn.is_subscribed("topic2").await);
        assert!(!conn.is_subscribed("topic3").await);

        assert_eq!(conn.subscription_count().await, 2);

        assert!(conn.unsubscribe("topic1").await);
        assert!(!conn.is_subscribed("topic1").await);
        assert_eq!(conn.subscription_count().await, 1);
    }

    #[tokio::test]
    async fn test_connection_groups() {
        let conn = Connection::new(ClientInfo::default());

        assert!(conn.join_group("group1").await);
        assert!(conn.is_in_group("group1").await);

        assert!(conn.leave_group("group1").await);
        assert!(!conn.is_in_group("group1").await);
    }

    #[tokio::test]
    async fn test_connection_metadata() {
        let conn = Connection::new(ClientInfo::default());

        conn.set_metadata("key1", json!("value1")).await;
        assert_eq!(conn.get_metadata("key1").await, Some(json!("value1")));
        assert_eq!(conn.get_metadata("key2").await, None);
    }

    #[tokio::test]
    async fn test_connection_is_stale() {
        let conn = Connection::new(ClientInfo::default());
        assert!(!conn.is_stale(60).await);
        // Note: Can't easily test stale without waiting
    }

    // ------------------------------------------------------------------------
    // Client Message Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_client_message_query() {
        let msg = ClientMessage::query("q1", "Hello world");
        assert_eq!(msg.message_type(), "query");

        match msg {
            ClientMessage::Query { id, text, agent, .. } => {
                assert_eq!(id, "q1");
                assert_eq!(text, "Hello world");
                assert!(agent.is_none());
            }
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn test_client_message_query_with_agent() {
        let msg = ClientMessage::query_with_agent("q1", "Hello", "assistant");

        match msg {
            ClientMessage::Query { agent, .. } => {
                assert_eq!(agent, Some("assistant".to_string()));
            }
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn test_client_message_subscribe() {
        let msg = ClientMessage::subscribe(vec!["topic1".to_string(), "topic2".to_string()]);
        assert_eq!(msg.message_type(), "subscribe");

        match msg {
            ClientMessage::Subscribe { topics } => {
                assert_eq!(topics.len(), 2);
            }
            _ => panic!("Expected Subscribe"),
        }
    }

    #[test]
    fn test_client_message_ping() {
        let msg = ClientMessage::ping();
        assert_eq!(msg.message_type(), "ping");
    }

    #[test]
    fn test_client_message_cancel_query() {
        let msg = ClientMessage::cancel_query("q123");
        assert_eq!(msg.message_type(), "cancel_query");
    }

    #[test]
    fn test_client_message_serialization() {
        let msg = ClientMessage::query("q1", "test");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"query\""));
        assert!(json.contains("\"id\":\"q1\""));
    }

    #[test]
    fn test_client_message_deserialization() {
        let json = r#"{"type":"ping"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::Ping));
    }

    // ------------------------------------------------------------------------
    // Server Message Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_server_message_query_start() {
        let msg = ServerMessage::query_start("q1");
        assert_eq!(msg.message_type(), "query_start");
    }

    #[test]
    fn test_server_message_query_chunk() {
        let msg = ServerMessage::query_chunk("q1", "Hello", false);
        assert_eq!(msg.message_type(), "query_chunk");

        match msg {
            ServerMessage::QueryChunk { id, chunk, done } => {
                assert_eq!(id, "q1");
                assert_eq!(chunk, "Hello");
                assert!(!done);
            }
            _ => panic!("Expected QueryChunk"),
        }
    }

    #[test]
    fn test_server_message_query_complete() {
        let result = QueryResult::new("response", 100)
            .with_token_count(50)
            .with_agent("assistant");
        let msg = ServerMessage::query_complete("q1", result);
        assert_eq!(msg.message_type(), "query_complete");
    }

    #[test]
    fn test_server_message_query_error() {
        let msg = ServerMessage::query_error("q1", "Something failed");
        assert!(msg.is_error());

        let msg_with_code = ServerMessage::query_error_with_code("q1", "Error", "E001");
        match msg_with_code {
            ServerMessage::QueryError { code, .. } => {
                assert_eq!(code, Some("E001".to_string()));
            }
            _ => panic!("Expected QueryError"),
        }
    }

    #[test]
    fn test_server_message_event() {
        let msg = ServerMessage::event("updates", json!({"version": "1.0"}));
        assert_eq!(msg.message_type(), "event");

        match msg {
            ServerMessage::Event { topic, data, .. } => {
                assert_eq!(topic, "updates");
                assert_eq!(data["version"], "1.0");
            }
            _ => panic!("Expected Event"),
        }
    }

    #[test]
    fn test_server_message_pong() {
        let msg = ServerMessage::pong();
        assert_eq!(msg.message_type(), "pong");
    }

    #[test]
    fn test_server_message_subscribed() {
        let msg = ServerMessage::subscribed(vec!["topic1".to_string()]);
        assert_eq!(msg.message_type(), "subscribed");
    }

    #[test]
    fn test_server_message_connected() {
        let msg = ServerMessage::connected("conn-123".to_string(), "1.0.0");
        assert_eq!(msg.message_type(), "connected");
    }

    #[test]
    fn test_server_message_reconnect_hint() {
        let msg = ServerMessage::reconnect_hint("Maintenance", 30);
        assert_eq!(msg.message_type(), "reconnect_hint");
    }

    #[test]
    fn test_server_message_is_error() {
        assert!(ServerMessage::query_error("q1", "err").is_error());
        assert!(ServerMessage::error("err").is_error());
        assert!(!ServerMessage::pong().is_error());
    }

    #[test]
    fn test_server_message_serialization() {
        let msg = ServerMessage::pong();
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"pong\""));
    }

    // ------------------------------------------------------------------------
    // Query Result Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_query_result_new() {
        let result = QueryResult::new("Hello world", 150);
        assert_eq!(result.response, "Hello world");
        assert_eq!(result.processing_time_ms, 150);
        assert_eq!(result.token_count, 0);
    }

    #[test]
    fn test_query_result_builder() {
        let result = QueryResult::new("response", 100)
            .with_token_count(50)
            .with_agent("gpt-4")
            .with_metadata("model", json!("gpt-4-turbo"));

        assert_eq!(result.token_count, 50);
        assert_eq!(result.agent, Some("gpt-4".to_string()));
        assert_eq!(result.metadata.get("model"), Some(&json!("gpt-4-turbo")));
    }

    // ------------------------------------------------------------------------
    // WebSocket Stats Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_stats_new() {
        let stats = WebSocketStats::new();
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_stats_record_connection() {
        let mut stats = WebSocketStats::new();
        stats.record_connection();
        stats.record_connection();
        assert_eq!(stats.total_connections, 2);
        assert_eq!(stats.active_connections, 2);
    }

    #[test]
    fn test_stats_record_disconnection() {
        let mut stats = WebSocketStats::new();
        stats.record_connection();
        stats.record_connection();
        stats.record_disconnection();
        assert_eq!(stats.active_connections, 1);
    }

    #[test]
    fn test_stats_record_messages() {
        let mut stats = WebSocketStats::new();
        stats.record_message_received(100);
        stats.record_message_sent(200);
        assert_eq!(stats.messages_received, 1);
        assert_eq!(stats.messages_sent, 1);
        assert_eq!(stats.bytes_received, 100);
        assert_eq!(stats.bytes_sent, 200);
    }

    #[test]
    fn test_stats_record_query() {
        let mut stats = WebSocketStats::new();
        stats.record_query();
        stats.record_query_error();
        assert_eq!(stats.queries_processed, 1);
        assert_eq!(stats.query_errors, 1);
    }

    // ------------------------------------------------------------------------
    // WebSocket Server Tests
    // ------------------------------------------------------------------------

    #[tokio::test]
    async fn test_server_new() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());
        assert_eq!(server.connection_count().await, 0);
        assert!(!server.is_at_capacity().await);
        assert!(!server.is_shutdown().await);
    }

    #[tokio::test]
    async fn test_server_with_defaults() {
        let server = WebSocketServer::with_defaults();
        assert_eq!(server.config().max_connections, 1000);
    }

    #[tokio::test]
    async fn test_server_accept_connection() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let conn = server.accept_connection(ClientInfo::default()).await.unwrap();
        assert_eq!(conn.get_state().await, ConnectionState::Connected);
        assert_eq!(server.connection_count().await, 1);

        let stats = server.stats().await;
        assert_eq!(stats.total_connections, 1);
        assert_eq!(stats.active_connections, 1);
    }

    #[tokio::test]
    async fn test_server_accept_connection_with_id() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let conn = server
            .accept_connection_with_id("my-conn", ClientInfo::default())
            .await
            .unwrap();
        assert_eq!(conn.id, "my-conn");

        // Should fail for duplicate ID
        let result = server
            .accept_connection_with_id("my-conn", ClientInfo::default())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_server_capacity_limit() {
        let config = WebSocketConfig {
            max_connections: 2,
            ..WebSocketConfig::minimal()
        };
        let server = WebSocketServer::new(config);

        server.accept_connection(ClientInfo::default()).await.unwrap();
        server.accept_connection(ClientInfo::default()).await.unwrap();

        assert!(server.is_at_capacity().await);

        let result = server.accept_connection(ClientInfo::default()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_server_get_connection() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let conn = server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();

        let retrieved = server.get_connection("conn-1").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, conn.id);

        let not_found = server.get_connection("nonexistent").await;
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_server_close_connection() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();

        assert!(server.close_connection("conn-1").await);
        assert_eq!(server.connection_count().await, 0);

        assert!(!server.close_connection("conn-1").await); // Already closed

        let stats = server.stats().await;
        assert_eq!(stats.active_connections, 0);
    }

    #[tokio::test]
    async fn test_server_subscriptions() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let conn = server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();

        let subscribed = server
            .subscribe("conn-1", vec!["topic1".to_string(), "topic2".to_string()])
            .await
            .unwrap();
        assert_eq!(subscribed.len(), 2);

        assert!(conn.is_subscribed("topic1").await);
        assert!(conn.is_subscribed("topic2").await);

        let subscribers = server.get_topic_subscribers("topic1").await;
        assert_eq!(subscribers.len(), 1);
        assert!(subscribers.contains(&"conn-1".to_string()));

        let topics = server.get_topics().await;
        assert_eq!(topics.len(), 2);

        let unsubscribed = server
            .unsubscribe("conn-1", vec!["topic1".to_string()])
            .await
            .unwrap();
        assert_eq!(unsubscribed.len(), 1);
        assert!(!conn.is_subscribed("topic1").await);
    }

    #[tokio::test]
    async fn test_server_subscription_limit() {
        let config = WebSocketConfig {
            max_subscriptions_per_connection: 2,
            ..WebSocketConfig::minimal()
        };
        let server = WebSocketServer::new(config);

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();

        let subscribed = server
            .subscribe(
                "conn-1",
                vec![
                    "t1".to_string(),
                    "t2".to_string(),
                    "t3".to_string(),
                ],
            )
            .await
            .unwrap();
        assert_eq!(subscribed.len(), 2); // Limited to 2
    }

    #[tokio::test]
    async fn test_server_groups() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let conn = server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();

        server.join_group("conn-1", "group1").await.unwrap();
        assert!(conn.is_in_group("group1").await);

        let members = server.get_group_members("group1").await;
        assert_eq!(members.len(), 1);

        let groups = server.get_groups().await;
        assert_eq!(groups.len(), 1);

        server.leave_group("conn-1", "group1").await.unwrap();
        assert!(!conn.is_in_group("group1").await);
    }

    #[tokio::test]
    async fn test_server_handle_ping() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();

        let response = server
            .handle_message("conn-1", ClientMessage::Ping)
            .await
            .unwrap();
        assert!(matches!(response, Some(ServerMessage::Pong)));

        let stats = server.stats().await;
        assert_eq!(stats.pings_received, 1);
    }

    #[tokio::test]
    async fn test_server_handle_subscribe() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();

        let response = server
            .handle_message(
                "conn-1",
                ClientMessage::Subscribe {
                    topics: vec!["topic1".to_string()],
                },
            )
            .await
            .unwrap();

        assert!(matches!(response, Some(ServerMessage::Subscribed { .. })));
    }

    #[tokio::test]
    async fn test_server_handle_query_no_handler() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();

        let response = server
            .handle_message(
                "conn-1",
                ClientMessage::Query {
                    id: "q1".to_string(),
                    text: "Hello".to_string(),
                    agent: None,
                    params: HashMap::new(),
                },
            )
            .await
            .unwrap();

        match response {
            Some(ServerMessage::QueryError { error, .. }) => {
                assert!(error.contains("No query handler"));
            }
            _ => panic!("Expected QueryError"),
        }
    }

    #[tokio::test]
    async fn test_server_broadcast() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server.accept_connection(ClientInfo::default()).await.unwrap();
        server.accept_connection(ClientInfo::default()).await.unwrap();

        let count = server.broadcast(ServerMessage::pong()).await;
        assert_eq!(count, 2);

        let stats = server.stats().await;
        assert_eq!(stats.broadcasts_sent, 1);
    }

    #[tokio::test]
    async fn test_server_publish() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();
        server
            .subscribe("conn-1", vec!["updates".to_string()])
            .await
            .unwrap();

        let count = server.publish("updates", json!({"msg": "hello"})).await;
        assert_eq!(count, 1);

        let stats = server.stats().await;
        assert_eq!(stats.events_published, 1);
    }

    #[tokio::test]
    async fn test_server_shutdown() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server.accept_connection(ClientInfo::default()).await.unwrap();
        server.accept_connection(ClientInfo::default()).await.unwrap();

        assert!(!server.is_shutdown().await);

        server.shutdown().await;

        assert!(server.is_shutdown().await);
        assert_eq!(server.connection_count().await, 0);

        // Cannot accept new connections after shutdown
        let result = server.accept_connection(ClientInfo::default()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_server_reset() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server.accept_connection(ClientInfo::default()).await.unwrap();
        server.shutdown().await;

        server.reset().await;

        assert!(!server.is_shutdown().await);
        assert_eq!(server.connection_count().await, 0);

        // Can accept connections again
        let result = server.accept_connection(ClientInfo::default()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_server_close_cleans_up_subscriptions() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();
        server
            .subscribe("conn-1", vec!["topic1".to_string()])
            .await
            .unwrap();

        assert_eq!(server.get_topic_subscribers("topic1").await.len(), 1);

        server.close_connection("conn-1").await;

        assert_eq!(server.get_topic_subscribers("topic1").await.len(), 0);
        assert_eq!(server.get_topics().await.len(), 0);
    }

    #[tokio::test]
    async fn test_server_close_cleans_up_groups() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();
        server.join_group("conn-1", "group1").await.unwrap();

        assert_eq!(server.get_group_members("group1").await.len(), 1);

        server.close_connection("conn-1").await;

        assert_eq!(server.get_group_members("group1").await.len(), 0);
        assert_eq!(server.get_groups().await.len(), 0);
    }

    // ------------------------------------------------------------------------
    // Builder Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let server = WebSocketServerBuilder::new().build();
        assert_eq!(server.config().max_connections, 1000);
    }

    #[test]
    fn test_builder_with_config() {
        let config = WebSocketConfig::minimal();
        let server = WebSocketServerBuilder::new().config(config).build();
        assert_eq!(server.config().max_connections, 10);
    }

    #[test]
    fn test_builder_custom_values() {
        let server = WebSocketServerBuilder::new()
            .max_connections(500)
            .heartbeat_interval_secs(60)
            .connection_timeout_secs(120)
            .max_message_size(2 * 1024 * 1024)
            .ping_interval_secs(20)
            .enable_compression(false)
            .build();

        assert_eq!(server.config().max_connections, 500);
        assert_eq!(server.config().heartbeat_interval_secs, 60);
        assert_eq!(server.config().connection_timeout_secs, 120);
        assert_eq!(server.config().max_message_size, 2 * 1024 * 1024);
        assert_eq!(server.config().ping_interval_secs, 20);
        assert!(!server.config().enable_compression);
    }

    // ------------------------------------------------------------------------
    // Query Handler Tests
    // ------------------------------------------------------------------------

    #[tokio::test]
    async fn test_server_query_handler_registration() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        assert!(!server.has_query_handler().await);

        server
            .register_query_handler(|_request, _tx| {
                Box::pin(async move { Ok(()) })
            })
            .await;

        assert!(server.has_query_handler().await);
    }

    #[tokio::test]
    async fn test_server_cancel_query() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        // No pending query
        assert!(!server.cancel_query("nonexistent").await);
    }

    // ------------------------------------------------------------------------
    // Edge Cases and Error Handling
    // ------------------------------------------------------------------------

    #[tokio::test]
    async fn test_subscribe_nonexistent_connection() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let result = server
            .subscribe("nonexistent", vec!["topic".to_string()])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_unsubscribe_nonexistent_connection() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let result = server
            .unsubscribe("nonexistent", vec!["topic".to_string()])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_join_group_nonexistent_connection() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let result = server.join_group("nonexistent", "group").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_leave_group_nonexistent_connection() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let result = server.leave_group("nonexistent", "group").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_multiple_connections_same_topic() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();
        server
            .accept_connection_with_id("conn-2", ClientInfo::default())
            .await
            .unwrap();

        server
            .subscribe("conn-1", vec!["topic".to_string()])
            .await
            .unwrap();
        server
            .subscribe("conn-2", vec!["topic".to_string()])
            .await
            .unwrap();

        let subscribers = server.get_topic_subscribers("topic").await;
        assert_eq!(subscribers.len(), 2);
    }

    #[tokio::test]
    async fn test_connection_ids() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();
        server
            .accept_connection_with_id("conn-2", ClientInfo::default())
            .await
            .unwrap();

        let ids = server.get_connection_ids().await;
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"conn-1".to_string()));
        assert!(ids.contains(&"conn-2".to_string()));
    }

    #[tokio::test]
    async fn test_send_to_group() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();
        server.join_group("conn-1", "group1").await.unwrap();

        let count = server
            .send_to_group("group1", ServerMessage::pong())
            .await;
        assert_eq!(count, 1);

        let count_empty = server
            .send_to_group("nonexistent", ServerMessage::pong())
            .await;
        assert_eq!(count_empty, 0);
    }

    #[tokio::test]
    async fn test_broadcast_receiver() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        let mut receiver = server.subscribe_broadcast();

        server.broadcast(ServerMessage::pong()).await;

        let msg = receiver.try_recv();
        assert!(msg.is_ok());
        assert!(matches!(msg.unwrap(), ServerMessage::Pong));
    }

    // ------------------------------------------------------------------------
    // Serialization Round-trip Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_client_message_roundtrip() {
        let messages = vec![
            ClientMessage::query("q1", "test"),
            ClientMessage::subscribe(vec!["topic".to_string()]),
            ClientMessage::ping(),
            ClientMessage::cancel_query("q2"),
            ClientMessage::Authenticate { token: "abc".to_string() },
        ];

        for msg in messages {
            let json = serde_json::to_string(&msg).unwrap();
            let deserialized: ClientMessage = serde_json::from_str(&json).unwrap();
            assert_eq!(msg.message_type(), deserialized.message_type());
        }
    }

    #[test]
    fn test_server_message_roundtrip() {
        let messages = vec![
            ServerMessage::query_start("q1"),
            ServerMessage::query_chunk("q1", "chunk", false),
            ServerMessage::pong(),
            ServerMessage::error("test error"),
            ServerMessage::connected("conn-1".to_string(), "1.0"),
        ];

        for msg in messages {
            let json = serde_json::to_string(&msg).unwrap();
            let deserialized: ServerMessage = serde_json::from_str(&json).unwrap();
            assert_eq!(msg.message_type(), deserialized.message_type());
        }
    }

    #[test]
    fn test_query_result_roundtrip() {
        let result = QueryResult::new("response", 100)
            .with_token_count(50)
            .with_agent("gpt-4");

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: QueryResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.response, deserialized.response);
        assert_eq!(result.processing_time_ms, deserialized.processing_time_ms);
        assert_eq!(result.token_count, deserialized.token_count);
    }

    // ------------------------------------------------------------------------
    // Authentication Handling
    // ------------------------------------------------------------------------

    #[tokio::test]
    async fn test_handle_authenticate() {
        let server = WebSocketServer::new(WebSocketConfig::minimal());

        server
            .accept_connection_with_id("conn-1", ClientInfo::default())
            .await
            .unwrap();

        let response = server
            .handle_message(
                "conn-1",
                ClientMessage::Authenticate { token: "test-token".to_string() },
            )
            .await
            .unwrap();

        match response {
            Some(ServerMessage::Authenticated { success, .. }) => {
                assert!(success);
            }
            _ => panic!("Expected Authenticated"),
        }

        // Verify token was stored
        let conn = server.get_connection("conn-1").await.unwrap();
        let token = conn.get_metadata("auth_token").await;
        assert_eq!(token, Some(json!("test-token")));
    }
}
