//! # REST API Server for Project Panpsychism
//!
//! HTTP API server for external integrations with OpenAPI-style endpoints.
//! This module provides a self-contained, educational HTTP server implementation
//! without external web framework dependencies.
//!
//! ## Philosophy
//!
//! In the Spinoza framework, the API embodies the bridge between worlds:
//!
//! - **CONATUS**: Self-preservation through proper error handling and rate limiting
//! - **RATIO**: Logical routing and structured request/response handling
//! - **NATURA**: Natural RESTful design aligned with HTTP semantics
//! - **LAETITIA**: Joy through clear documentation and predictable behavior
//!
//! ## Architecture
//!
//! ```text
//! +------------------+     +------------------+     +------------------+
//! |   HTTP Request   | --> |    Middleware    | --> |     Router       |
//! |   (Method, Path) |     |  (Auth, CORS)    |     | (Path Matching)  |
//! +------------------+     +------------------+     +------------------+
//!                                                          |
//!                                                          v
//!                                                  +------------------+
//!                                                  |   Route Handler  |
//!                                                  |  (Business Logic)|
//!                                                  +------------------+
//!                                                          |
//!                                                          v
//!                                                  +------------------+
//!                                                  |   HTTP Response  |
//!                                                  | (Status, Headers)|
//!                                                  +------------------+
//! ```
//!
//! ## Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | /api/v1/query | Process a query |
//! | POST | /api/v1/agents/{agent_id}/invoke | Invoke specific agent |
//! | GET | /api/v1/agents | List all agents |
//! | GET | /api/v1/agents/{agent_id}/status | Get agent status |
//! | GET | /api/v1/health | Health check |
//! | GET | /api/v1/metrics | Performance metrics |
//!
//! ## Example
//!
//! ```rust,ignore
//! use panpsychism::api::{ApiServer, ApiConfig, Router};
//!
//! #[tokio::main]
//! async fn main() -> panpsychism::Result<()> {
//!     let server = ApiServer::builder()
//!         .host("127.0.0.1")
//!         .port(8080)
//!         .cors_origin("http://localhost:3000")
//!         .build()?;
//!
//!     server.run().await
//! }
//! ```

use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use uuid::Uuid;

// =============================================================================
// TYPE ALIASES
// =============================================================================

/// Type alias for request IDs.
pub type RequestId = String;

/// Type alias for header maps.
pub type Headers = HashMap<String, String>;

/// Type alias for query parameters.
pub type QueryParams = HashMap<String, String>;

/// Type alias for path parameters.
pub type PathParams = HashMap<String, String>;

/// Type alias for async route handler functions.
pub type RouteHandler = Arc<
    dyn Fn(Request, Arc<ApiState>) -> Pin<Box<dyn Future<Output = Response> + Send>>
        + Send
        + Sync,
>;

// =============================================================================
// HTTP METHOD
// =============================================================================

/// HTTP methods supported by the API server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Method {
    /// GET method for retrieving resources.
    GET,
    /// POST method for creating resources.
    POST,
    /// PUT method for updating resources.
    PUT,
    /// DELETE method for removing resources.
    DELETE,
    /// PATCH method for partial updates.
    PATCH,
    /// HEAD method for metadata retrieval.
    HEAD,
    /// OPTIONS method for CORS preflight.
    OPTIONS,
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Method::GET => write!(f, "GET"),
            Method::POST => write!(f, "POST"),
            Method::PUT => write!(f, "PUT"),
            Method::DELETE => write!(f, "DELETE"),
            Method::PATCH => write!(f, "PATCH"),
            Method::HEAD => write!(f, "HEAD"),
            Method::OPTIONS => write!(f, "OPTIONS"),
        }
    }
}

impl Method {
    /// Parse a method from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Some(Method::GET),
            "POST" => Some(Method::POST),
            "PUT" => Some(Method::PUT),
            "DELETE" => Some(Method::DELETE),
            "PATCH" => Some(Method::PATCH),
            "HEAD" => Some(Method::HEAD),
            "OPTIONS" => Some(Method::OPTIONS),
            _ => None,
        }
    }

    /// Check if this method typically has a request body.
    pub fn has_body(&self) -> bool {
        matches!(self, Method::POST | Method::PUT | Method::PATCH)
    }

    /// Check if this method is safe (no side effects).
    pub fn is_safe(&self) -> bool {
        matches!(self, Method::GET | Method::HEAD | Method::OPTIONS)
    }

    /// Check if this method is idempotent.
    pub fn is_idempotent(&self) -> bool {
        matches!(
            self,
            Method::GET | Method::HEAD | Method::OPTIONS | Method::PUT | Method::DELETE
        )
    }
}

// =============================================================================
// HTTP STATUS
// =============================================================================

/// HTTP status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Status(pub u16);

impl Status {
    // 2xx Success
    /// 200 OK
    pub const OK: Status = Status(200);
    /// 201 Created
    pub const CREATED: Status = Status(201);
    /// 202 Accepted
    pub const ACCEPTED: Status = Status(202);
    /// 204 No Content
    pub const NO_CONTENT: Status = Status(204);

    // 3xx Redirection
    /// 301 Moved Permanently
    pub const MOVED_PERMANENTLY: Status = Status(301);
    /// 302 Found
    pub const FOUND: Status = Status(302);
    /// 304 Not Modified
    pub const NOT_MODIFIED: Status = Status(304);

    // 4xx Client Errors
    /// 400 Bad Request
    pub const BAD_REQUEST: Status = Status(400);
    /// 401 Unauthorized
    pub const UNAUTHORIZED: Status = Status(401);
    /// 403 Forbidden
    pub const FORBIDDEN: Status = Status(403);
    /// 404 Not Found
    pub const NOT_FOUND: Status = Status(404);
    /// 405 Method Not Allowed
    pub const METHOD_NOT_ALLOWED: Status = Status(405);
    /// 408 Request Timeout
    pub const REQUEST_TIMEOUT: Status = Status(408);
    /// 409 Conflict
    pub const CONFLICT: Status = Status(409);
    /// 413 Payload Too Large
    pub const PAYLOAD_TOO_LARGE: Status = Status(413);
    /// 422 Unprocessable Entity
    pub const UNPROCESSABLE_ENTITY: Status = Status(422);
    /// 429 Too Many Requests
    pub const TOO_MANY_REQUESTS: Status = Status(429);

    // 5xx Server Errors
    /// 500 Internal Server Error
    pub const INTERNAL_SERVER_ERROR: Status = Status(500);
    /// 501 Not Implemented
    pub const NOT_IMPLEMENTED: Status = Status(501);
    /// 502 Bad Gateway
    pub const BAD_GATEWAY: Status = Status(502);
    /// 503 Service Unavailable
    pub const SERVICE_UNAVAILABLE: Status = Status(503);
    /// 504 Gateway Timeout
    pub const GATEWAY_TIMEOUT: Status = Status(504);

    /// Get the status code as a u16.
    pub fn code(&self) -> u16 {
        self.0
    }

    /// Get the reason phrase for this status code.
    pub fn reason(&self) -> &'static str {
        match self.0 {
            200 => "OK",
            201 => "Created",
            202 => "Accepted",
            204 => "No Content",
            301 => "Moved Permanently",
            302 => "Found",
            304 => "Not Modified",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            408 => "Request Timeout",
            409 => "Conflict",
            413 => "Payload Too Large",
            422 => "Unprocessable Entity",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            501 => "Not Implemented",
            502 => "Bad Gateway",
            503 => "Service Unavailable",
            504 => "Gateway Timeout",
            _ => "Unknown",
        }
    }

    /// Check if this is a success status (2xx).
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.0)
    }

    /// Check if this is a client error (4xx).
    pub fn is_client_error(&self) -> bool {
        (400..500).contains(&self.0)
    }

    /// Check if this is a server error (5xx).
    pub fn is_server_error(&self) -> bool {
        (500..600).contains(&self.0)
    }

    /// Check if this is an error (4xx or 5xx).
    pub fn is_error(&self) -> bool {
        self.is_client_error() || self.is_server_error()
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.0, self.reason())
    }
}

// =============================================================================
// REQUEST
// =============================================================================

/// HTTP request representation.
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request identifier for tracing.
    pub id: RequestId,
    /// HTTP method.
    pub method: Method,
    /// Request path (e.g., "/api/v1/agents").
    pub path: String,
    /// Request headers.
    pub headers: Headers,
    /// Query parameters.
    pub query_params: QueryParams,
    /// Path parameters extracted from route.
    pub path_params: PathParams,
    /// Request body as bytes.
    pub body: Vec<u8>,
    /// Request timestamp.
    pub timestamp: DateTime<Utc>,
    /// Client IP address (if available).
    pub client_ip: Option<String>,
}

impl Request {
    /// Create a new request with the given method and path.
    pub fn new(method: Method, path: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            method,
            path: path.into(),
            headers: HashMap::new(),
            query_params: HashMap::new(),
            path_params: HashMap::new(),
            body: Vec::new(),
            timestamp: Utc::now(),
            client_ip: None,
        }
    }

    /// Add a header to the request.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into().to_lowercase(), value.into());
        self
    }

    /// Add a query parameter.
    pub fn with_query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.insert(key.into(), value.into());
        self
    }

    /// Set the request body.
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = body;
        self
    }

    /// Set the request body as JSON.
    pub fn with_json<T: Serialize>(mut self, value: &T) -> Result<Self> {
        self.body = serde_json::to_vec(value)?;
        self.headers
            .insert("content-type".to_string(), "application/json".to_string());
        Ok(self)
    }

    /// Set the client IP.
    pub fn with_client_ip(mut self, ip: impl Into<String>) -> Self {
        self.client_ip = Some(ip.into());
        self
    }

    /// Get a header value (case-insensitive).
    pub fn header(&self, key: &str) -> Option<&str> {
        self.headers.get(&key.to_lowercase()).map(|s| s.as_str())
    }

    /// Get a query parameter.
    pub fn query(&self, key: &str) -> Option<&str> {
        self.query_params.get(key).map(|s| s.as_str())
    }

    /// Get a path parameter.
    pub fn param(&self, key: &str) -> Option<&str> {
        self.path_params.get(key).map(|s| s.as_str())
    }

    /// Get the body as a string (UTF-8).
    pub fn body_string(&self) -> Result<String> {
        String::from_utf8(self.body.clone()).map_err(|e| crate::Error::internal(e.to_string()))
    }

    /// Parse the body as JSON.
    pub fn body_json<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        serde_json::from_slice(&self.body).map_err(|e| e.into())
    }

    /// Get the content type header.
    pub fn content_type(&self) -> Option<&str> {
        self.header("content-type")
    }

    /// Check if the request accepts JSON.
    pub fn accepts_json(&self) -> bool {
        self.header("accept")
            .map(|a| a.contains("application/json") || a.contains("*/*"))
            .unwrap_or(true)
    }

    /// Get the content length.
    pub fn content_length(&self) -> Option<usize> {
        self.header("content-length")
            .and_then(|s| s.parse().ok())
    }

    /// Get the authorization header.
    pub fn authorization(&self) -> Option<&str> {
        self.header("authorization")
    }

    /// Extract bearer token from authorization header.
    pub fn bearer_token(&self) -> Option<&str> {
        self.authorization().and_then(|auth| {
            if auth.to_lowercase().starts_with("bearer ") {
                Some(&auth[7..])
            } else {
                None
            }
        })
    }
}

impl Default for Request {
    fn default() -> Self {
        Self::new(Method::GET, "/")
    }
}

// =============================================================================
// RESPONSE
// =============================================================================

/// HTTP response representation.
#[derive(Debug, Clone)]
pub struct Response {
    /// HTTP status.
    pub status: Status,
    /// Response headers.
    pub headers: Headers,
    /// Response body.
    pub body: Vec<u8>,
    /// Response timestamp.
    pub timestamp: DateTime<Utc>,
}

impl Response {
    /// Create a new response with the given status.
    pub fn new(status: Status) -> Self {
        Self {
            status,
            headers: HashMap::new(),
            body: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    /// Create a 200 OK response.
    pub fn ok() -> Self {
        Self::new(Status::OK)
    }

    /// Create a 201 Created response.
    pub fn created() -> Self {
        Self::new(Status::CREATED)
    }

    /// Create a 204 No Content response.
    pub fn no_content() -> Self {
        Self::new(Status::NO_CONTENT)
    }

    /// Create a 400 Bad Request response.
    pub fn bad_request() -> Self {
        Self::new(Status::BAD_REQUEST)
    }

    /// Create a 401 Unauthorized response.
    pub fn unauthorized() -> Self {
        Self::new(Status::UNAUTHORIZED)
    }

    /// Create a 403 Forbidden response.
    pub fn forbidden() -> Self {
        Self::new(Status::FORBIDDEN)
    }

    /// Create a 404 Not Found response.
    pub fn not_found() -> Self {
        Self::new(Status::NOT_FOUND)
    }

    /// Create a 405 Method Not Allowed response.
    pub fn method_not_allowed() -> Self {
        Self::new(Status::METHOD_NOT_ALLOWED)
    }

    /// Create a 429 Too Many Requests response.
    pub fn too_many_requests() -> Self {
        Self::new(Status::TOO_MANY_REQUESTS)
    }

    /// Create a 500 Internal Server Error response.
    pub fn internal_error() -> Self {
        Self::new(Status::INTERNAL_SERVER_ERROR)
    }

    /// Create a 503 Service Unavailable response.
    pub fn service_unavailable() -> Self {
        Self::new(Status::SERVICE_UNAVAILABLE)
    }

    /// Add a header to the response.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into().to_lowercase(), value.into());
        self
    }

    /// Set the response body.
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = body;
        self
    }

    /// Set the response body as JSON.
    pub fn with_json<T: Serialize>(self, value: &T) -> Result<Self> {
        let body = serde_json::to_vec(value)?;
        Ok(self
            .with_header("content-type", "application/json")
            .with_body(body))
    }

    /// Set the response body as a string.
    pub fn with_text(self, text: impl Into<String>) -> Self {
        self.with_header("content-type", "text/plain")
            .with_body(text.into().into_bytes())
    }

    /// Get a header value.
    pub fn header(&self, key: &str) -> Option<&str> {
        self.headers.get(&key.to_lowercase()).map(|s| s.as_str())
    }

    /// Get the body as a string.
    pub fn body_string(&self) -> Result<String> {
        String::from_utf8(self.body.clone()).map_err(|e| crate::Error::internal(e.to_string()))
    }

    /// Parse the body as JSON.
    pub fn body_json<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        serde_json::from_slice(&self.body).map_err(|e| e.into())
    }

    /// Check if the response is successful.
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Check if the response is an error.
    pub fn is_error(&self) -> bool {
        self.status.is_error()
    }
}

impl Default for Response {
    fn default() -> Self {
        Self::ok()
    }
}

// =============================================================================
// API ERROR
// =============================================================================

/// Structured API error for consistent error responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// Error code for programmatic handling.
    pub code: String,
    /// Human-readable error message.
    pub message: String,
    /// HTTP status code.
    pub status: u16,
    /// Optional details for debugging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// Request ID for tracing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Timestamp of the error.
    pub timestamp: DateTime<Utc>,
}

impl ApiError {
    /// Create a new API error.
    pub fn new(code: impl Into<String>, message: impl Into<String>, status: Status) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            status: status.code(),
            details: None,
            request_id: None,
            timestamp: Utc::now(),
        }
    }

    /// Add details to the error.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    /// Add request ID for tracing.
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    /// Create a bad request error.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new("BAD_REQUEST", message, Status::BAD_REQUEST)
    }

    /// Create an unauthorized error.
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new("UNAUTHORIZED", message, Status::UNAUTHORIZED)
    }

    /// Create a forbidden error.
    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::new("FORBIDDEN", message, Status::FORBIDDEN)
    }

    /// Create a not found error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new("NOT_FOUND", message, Status::NOT_FOUND)
    }

    /// Create a method not allowed error.
    pub fn method_not_allowed(method: &str) -> Self {
        Self::new(
            "METHOD_NOT_ALLOWED",
            format!("Method {} not allowed", method),
            Status::METHOD_NOT_ALLOWED,
        )
    }

    /// Create a rate limit error.
    pub fn rate_limited(retry_after_secs: Option<u64>) -> Self {
        let mut err = Self::new(
            "RATE_LIMITED",
            "Too many requests",
            Status::TOO_MANY_REQUESTS,
        );
        if let Some(secs) = retry_after_secs {
            err.details = Some(serde_json::json!({ "retry_after_secs": secs }));
        }
        err
    }

    /// Create a validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::new("VALIDATION_ERROR", message, Status::UNPROCESSABLE_ENTITY)
    }

    /// Create an internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(
            "INTERNAL_ERROR",
            message,
            Status::INTERNAL_SERVER_ERROR,
        )
    }

    /// Convert to an HTTP response.
    pub fn into_response(self) -> Response {
        let status = Status(self.status);
        Response::new(status)
            .with_json(&self)
            .unwrap_or_else(|_| Response::new(status).with_text("Internal error"))
    }
}

// =============================================================================
// ROUTE
// =============================================================================

/// A route definition with method, path pattern, and handler.
#[derive(Clone)]
pub struct Route {
    /// HTTP method for this route.
    pub method: Method,
    /// Path pattern (supports {param} syntax).
    pub pattern: String,
    /// Path segments for matching.
    segments: Vec<PathSegment>,
    /// Route handler.
    pub handler: RouteHandler,
    /// Route name for documentation.
    pub name: Option<String>,
    /// Route description for documentation.
    pub description: Option<String>,
}

/// A segment of a path pattern.
#[derive(Debug, Clone, PartialEq)]
enum PathSegment {
    /// Literal segment (exact match).
    Literal(String),
    /// Parameter segment (captures value).
    Param(String),
    /// Wildcard segment (matches anything).
    Wildcard,
}

impl Route {
    /// Create a new route with the given method, pattern, and handler.
    pub fn new<F, Fut>(method: Method, pattern: impl Into<String>, handler: F) -> Self
    where
        F: Fn(Request, Arc<ApiState>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        let pattern = pattern.into();
        let segments = Self::parse_pattern(&pattern);

        Self {
            method,
            pattern,
            segments,
            handler: Arc::new(move |req, state| Box::pin(handler(req, state))),
            name: None,
            description: None,
        }
    }

    /// Set the route name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the route description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Parse a path pattern into segments.
    fn parse_pattern(pattern: &str) -> Vec<PathSegment> {
        pattern
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|segment| {
                if segment.starts_with('{') && segment.ends_with('}') {
                    let param_name = segment[1..segment.len() - 1].to_string();
                    PathSegment::Param(param_name)
                } else if segment == "*" {
                    PathSegment::Wildcard
                } else {
                    PathSegment::Literal(segment.to_string())
                }
            })
            .collect()
    }

    /// Match a path against this route's pattern.
    ///
    /// Returns extracted path parameters if matched.
    pub fn matches(&self, path: &str) -> Option<PathParams> {
        let path_segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        // Check for wildcard at end
        let has_wildcard = self
            .segments
            .last()
            .map(|s| matches!(s, PathSegment::Wildcard))
            .unwrap_or(false);

        if !has_wildcard && path_segments.len() != self.segments.len() {
            return None;
        }

        if has_wildcard && path_segments.len() < self.segments.len() - 1 {
            return None;
        }

        let mut params = PathParams::new();

        for (i, segment) in self.segments.iter().enumerate() {
            match segment {
                PathSegment::Literal(lit) => {
                    if path_segments.get(i) != Some(&lit.as_str()) {
                        return None;
                    }
                }
                PathSegment::Param(name) => {
                    let value = path_segments.get(i)?;
                    params.insert(name.clone(), (*value).to_string());
                }
                PathSegment::Wildcard => {
                    // Wildcard matches rest of path
                    break;
                }
            }
        }

        Some(params)
    }
}

impl std::fmt::Debug for Route {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Route")
            .field("method", &self.method)
            .field("pattern", &self.pattern)
            .field("name", &self.name)
            .finish()
    }
}

// =============================================================================
// MIDDLEWARE
// =============================================================================

/// Middleware trait for request/response processing.
pub trait Middleware: Send + Sync {
    /// Process the request before the handler.
    ///
    /// Return `Some(Response)` to short-circuit and skip the handler.
    /// Return `None` to continue to the handler.
    fn before(&self, request: &Request, state: &ApiState) -> Option<Response>;

    /// Process the response after the handler.
    fn after(&self, request: &Request, response: Response, state: &ApiState) -> Response;

    /// Get the middleware name for logging.
    fn name(&self) -> &'static str;
}

/// CORS middleware for cross-origin requests.
#[derive(Debug, Clone)]
pub struct CorsMiddleware {
    /// Allowed origins.
    pub allowed_origins: Vec<String>,
    /// Allowed methods.
    pub allowed_methods: Vec<Method>,
    /// Allowed headers.
    pub allowed_headers: Vec<String>,
    /// Max age for preflight cache (seconds).
    pub max_age_secs: u64,
}

impl Default for CorsMiddleware {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                Method::GET,
                Method::POST,
                Method::PUT,
                Method::DELETE,
                Method::PATCH,
                Method::OPTIONS,
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-Request-ID".to_string(),
            ],
            max_age_secs: 86400,
        }
    }
}

impl CorsMiddleware {
    /// Create a new CORS middleware with specific origins.
    pub fn new(origins: Vec<String>) -> Self {
        Self {
            allowed_origins: origins,
            ..Default::default()
        }
    }

    /// Check if an origin is allowed.
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.allowed_origins.contains(&"*".to_string())
            || self.allowed_origins.iter().any(|o| o == origin)
    }

    /// Get CORS headers.
    fn cors_headers(&self, origin: Option<&str>) -> Headers {
        let mut headers = Headers::new();

        let allowed_origin = if self.allowed_origins.contains(&"*".to_string()) {
            "*".to_string()
        } else {
            origin.unwrap_or("*").to_string()
        };

        headers.insert(
            "access-control-allow-origin".to_string(),
            allowed_origin,
        );
        headers.insert(
            "access-control-allow-methods".to_string(),
            self.allowed_methods
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        );
        headers.insert(
            "access-control-allow-headers".to_string(),
            self.allowed_headers.join(", "),
        );
        headers.insert(
            "access-control-max-age".to_string(),
            self.max_age_secs.to_string(),
        );

        headers
    }
}

impl Middleware for CorsMiddleware {
    fn before(&self, request: &Request, _state: &ApiState) -> Option<Response> {
        // Handle preflight OPTIONS requests
        if request.method == Method::OPTIONS {
            let origin = request.header("origin");
            let mut response = Response::no_content();
            for (k, v) in self.cors_headers(origin) {
                response = response.with_header(k, v);
            }
            return Some(response);
        }

        None
    }

    fn after(&self, request: &Request, mut response: Response, _state: &ApiState) -> Response {
        let origin = request.header("origin");
        for (k, v) in self.cors_headers(origin) {
            response.headers.insert(k, v);
        }
        response
    }

    fn name(&self) -> &'static str {
        "CorsMiddleware"
    }
}

/// Rate limiting middleware.
#[derive(Debug)]
pub struct RateLimitMiddleware {
    /// Maximum requests per window.
    pub max_requests: u32,
    /// Time window in seconds.
    pub window_secs: u64,
    /// Request counts per client.
    counts: RwLock<HashMap<String, (u32, Instant)>>,
}

impl Default for RateLimitMiddleware {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window_secs: 60,
            counts: RwLock::new(HashMap::new()),
        }
    }
}

impl RateLimitMiddleware {
    /// Create a new rate limit middleware.
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self {
            max_requests,
            window_secs,
            counts: RwLock::new(HashMap::new()),
        }
    }

    /// Get client identifier from request.
    fn get_client_id(&self, request: &Request) -> String {
        // Try X-Forwarded-For, then client_ip, then default
        request
            .header("x-forwarded-for")
            .map(|s| s.split(',').next().unwrap_or("unknown").trim().to_string())
            .or_else(|| request.client_ip.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Check and update rate limit for a client.
    fn check_rate(&self, client_id: &str) -> (bool, u32, u64) {
        let mut counts = self.counts.write().unwrap();
        let now = Instant::now();
        let window = Duration::from_secs(self.window_secs);

        let (count, window_start) = counts
            .entry(client_id.to_string())
            .or_insert((0, now));

        // Reset if window expired
        if now.duration_since(*window_start) >= window {
            *count = 0;
            *window_start = now;
        }

        let remaining = self.max_requests.saturating_sub(*count);
        let reset_secs = (window - now.duration_since(*window_start)).as_secs();

        if *count >= self.max_requests {
            (false, remaining, reset_secs)
        } else {
            *count += 1;
            (true, remaining.saturating_sub(1), reset_secs)
        }
    }
}

impl Middleware for RateLimitMiddleware {
    fn before(&self, request: &Request, _state: &ApiState) -> Option<Response> {
        let client_id = self.get_client_id(request);
        let (allowed, remaining, reset_secs) = self.check_rate(&client_id);

        if !allowed {
            let error = ApiError::rate_limited(Some(reset_secs));
            let mut response = error.into_response();
            response = response
                .with_header("x-ratelimit-limit", self.max_requests.to_string())
                .with_header("x-ratelimit-remaining", "0")
                .with_header("x-ratelimit-reset", reset_secs.to_string())
                .with_header("retry-after", reset_secs.to_string());
            return Some(response);
        }

        None
    }

    fn after(&self, request: &Request, response: Response, _state: &ApiState) -> Response {
        let client_id = self.get_client_id(request);
        let counts = self.counts.read().unwrap();

        if let Some((count, window_start)) = counts.get(&client_id) {
            let remaining = self.max_requests.saturating_sub(*count);
            let reset_secs =
                (Duration::from_secs(self.window_secs) - Instant::now().duration_since(*window_start))
                    .as_secs();

            return response
                .with_header("x-ratelimit-limit", self.max_requests.to_string())
                .with_header("x-ratelimit-remaining", remaining.to_string())
                .with_header("x-ratelimit-reset", reset_secs.to_string());
        }

        response
    }

    fn name(&self) -> &'static str {
        "RateLimitMiddleware"
    }
}

/// Request logging middleware.
#[derive(Debug, Default)]
pub struct LoggingMiddleware {
    /// Log request bodies.
    pub log_bodies: bool,
}

impl LoggingMiddleware {
    /// Create a new logging middleware.
    pub fn new(log_bodies: bool) -> Self {
        Self { log_bodies }
    }
}

impl Middleware for LoggingMiddleware {
    fn before(&self, _request: &Request, _state: &ApiState) -> Option<Response> {
        // Logging is handled externally; this is a placeholder
        None
    }

    fn after(&self, _request: &Request, response: Response, _state: &ApiState) -> Response {
        // Logging is handled externally; this is a placeholder
        response
    }

    fn name(&self) -> &'static str {
        "LoggingMiddleware"
    }
}

/// Authentication middleware.
#[derive(Debug)]
pub struct AuthMiddleware {
    /// Valid API keys.
    api_keys: Vec<String>,
    /// Paths that don't require authentication.
    public_paths: Vec<String>,
}

impl AuthMiddleware {
    /// Create a new auth middleware with API keys.
    pub fn new(api_keys: Vec<String>) -> Self {
        Self {
            api_keys,
            public_paths: vec!["/api/v1/health".to_string()],
        }
    }

    /// Add a public path that doesn't require authentication.
    pub fn with_public_path(mut self, path: impl Into<String>) -> Self {
        self.public_paths.push(path.into());
        self
    }

    /// Check if a path is public.
    fn is_public_path(&self, path: &str) -> bool {
        self.public_paths.iter().any(|p| path.starts_with(p))
    }

    /// Validate an API key.
    fn validate_key(&self, key: &str) -> bool {
        self.api_keys.iter().any(|k| k == key)
    }
}

impl Middleware for AuthMiddleware {
    fn before(&self, request: &Request, _state: &ApiState) -> Option<Response> {
        // Skip authentication for public paths
        if self.is_public_path(&request.path) {
            return None;
        }

        // Check for bearer token
        match request.bearer_token() {
            Some(token) if self.validate_key(token) => None,
            Some(_) => Some(ApiError::unauthorized("Invalid API key").into_response()),
            None => Some(ApiError::unauthorized("Missing API key").into_response()),
        }
    }

    fn after(&self, _request: &Request, response: Response, _state: &ApiState) -> Response {
        response
    }

    fn name(&self) -> &'static str {
        "AuthMiddleware"
    }
}

// =============================================================================
// ROUTER
// =============================================================================

/// HTTP router for matching requests to handlers.
#[derive(Default)]
pub struct Router {
    /// Registered routes.
    routes: Vec<Route>,
    /// Middleware chain.
    middleware: Vec<Box<dyn Middleware>>,
}

impl Router {
    /// Create a new router.
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            middleware: Vec::new(),
        }
    }

    /// Add a route to the router.
    pub fn route(mut self, route: Route) -> Self {
        self.routes.push(route);
        self
    }

    /// Add a GET route.
    pub fn get<F, Fut>(self, pattern: impl Into<String>, handler: F) -> Self
    where
        F: Fn(Request, Arc<ApiState>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        self.route(Route::new(Method::GET, pattern, handler))
    }

    /// Add a POST route.
    pub fn post<F, Fut>(self, pattern: impl Into<String>, handler: F) -> Self
    where
        F: Fn(Request, Arc<ApiState>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        self.route(Route::new(Method::POST, pattern, handler))
    }

    /// Add a PUT route.
    pub fn put<F, Fut>(self, pattern: impl Into<String>, handler: F) -> Self
    where
        F: Fn(Request, Arc<ApiState>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        self.route(Route::new(Method::PUT, pattern, handler))
    }

    /// Add a DELETE route.
    pub fn delete<F, Fut>(self, pattern: impl Into<String>, handler: F) -> Self
    where
        F: Fn(Request, Arc<ApiState>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        self.route(Route::new(Method::DELETE, pattern, handler))
    }

    /// Add middleware to the router.
    pub fn middleware<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middleware.push(Box::new(middleware));
        self
    }

    /// Find a matching route for a request.
    pub fn find_route(&self, method: Method, path: &str) -> Option<(&Route, PathParams)> {
        for route in &self.routes {
            if route.method == method {
                if let Some(params) = route.matches(path) {
                    return Some((route, params));
                }
            }
        }
        None
    }

    /// Check if a path has any matching routes (for 405 detection).
    pub fn path_has_routes(&self, path: &str) -> bool {
        self.routes.iter().any(|r| r.matches(path).is_some())
    }

    /// Get all routes for documentation.
    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    /// Get all middleware names.
    pub fn middleware_names(&self) -> Vec<&'static str> {
        self.middleware.iter().map(|m| m.name()).collect()
    }

    /// Process a request through the router.
    pub async fn handle(&self, mut request: Request, state: Arc<ApiState>) -> Response {
        // Run before middleware
        for mw in &self.middleware {
            if let Some(response) = mw.before(&request, &state) {
                // Short-circuit, but still run after middleware
                let mut final_response = response;
                for after_mw in &self.middleware {
                    final_response = after_mw.after(&request, final_response, &state);
                }
                return final_response;
            }
        }

        // Find and execute route
        let response = match self.find_route(request.method, &request.path) {
            Some((route, params)) => {
                request.path_params = params;
                (route.handler)(request.clone(), state.clone()).await
            }
            None => {
                if self.path_has_routes(&request.path) {
                    ApiError::method_not_allowed(&request.method.to_string()).into_response()
                } else {
                    ApiError::not_found(format!("Route not found: {}", request.path))
                        .into_response()
                }
            }
        };

        // Run after middleware
        let mut final_response = response;
        for mw in &self.middleware {
            final_response = mw.after(&request, final_response, &state);
        }

        final_response
    }
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("routes", &self.routes.len())
            .field("middleware", &self.middleware_names())
            .finish()
    }
}

// =============================================================================
// API STATE
// =============================================================================

/// Shared state for API handlers.
#[derive(Debug, Default)]
pub struct ApiState {
    /// Registered agents and their status.
    pub agents: RwLock<HashMap<String, AgentInfo>>,
    /// Server metrics.
    pub metrics: RwLock<ServerMetrics>,
    /// Server start time.
    pub started_at: DateTime<Utc>,
}

/// Information about a registered agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent ID.
    pub id: String,
    /// Agent name.
    pub name: String,
    /// Agent status.
    pub status: AgentStatus,
    /// Last activity timestamp.
    pub last_activity: DateTime<Utc>,
    /// Agent metadata.
    pub metadata: HashMap<String, String>,
}

/// Agent status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is active and processing.
    Active,
    /// Agent is idle.
    Idle,
    /// Agent is processing a request.
    Busy,
    /// Agent has encountered an error.
    Error,
    /// Agent is offline.
    Offline,
}

impl Default for AgentStatus {
    fn default() -> Self {
        Self::Idle
    }
}

/// Server metrics.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ServerMetrics {
    /// Total requests processed.
    pub total_requests: u64,
    /// Successful requests.
    pub successful_requests: u64,
    /// Failed requests.
    pub failed_requests: u64,
    /// Total response time in milliseconds.
    pub total_response_time_ms: u64,
    /// Active connections.
    pub active_connections: u32,
}

impl ServerMetrics {
    /// Record a request.
    pub fn record_request(&mut self, success: bool, duration_ms: u64) {
        self.total_requests += 1;
        self.total_response_time_ms += duration_ms;
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }
    }

    /// Get average response time in milliseconds.
    pub fn avg_response_time_ms(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_response_time_ms as f64 / self.total_requests as f64
        }
    }

    /// Get success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            100.0
        } else {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        }
    }
}

impl ApiState {
    /// Create a new API state.
    pub fn new() -> Self {
        Self {
            agents: RwLock::new(HashMap::new()),
            metrics: RwLock::new(ServerMetrics::default()),
            started_at: Utc::now(),
        }
    }

    /// Register an agent.
    pub fn register_agent(&self, id: impl Into<String>, name: impl Into<String>) {
        let id = id.into();
        let mut agents = self.agents.write().unwrap();
        agents.insert(
            id.clone(),
            AgentInfo {
                id,
                name: name.into(),
                status: AgentStatus::Idle,
                last_activity: Utc::now(),
                metadata: HashMap::new(),
            },
        );
    }

    /// Get an agent by ID.
    pub fn get_agent(&self, id: &str) -> Option<AgentInfo> {
        self.agents.read().unwrap().get(id).cloned()
    }

    /// List all agents.
    pub fn list_agents(&self) -> Vec<AgentInfo> {
        self.agents.read().unwrap().values().cloned().collect()
    }

    /// Update agent status.
    pub fn update_agent_status(&self, id: &str, status: AgentStatus) -> bool {
        if let Some(agent) = self.agents.write().unwrap().get_mut(id) {
            agent.status = status;
            agent.last_activity = Utc::now();
            true
        } else {
            false
        }
    }

    /// Record a request in metrics.
    pub fn record_request(&self, success: bool, duration_ms: u64) {
        self.metrics
            .write()
            .unwrap()
            .record_request(success, duration_ms);
    }

    /// Get server metrics.
    pub fn get_metrics(&self) -> ServerMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Get server uptime in seconds.
    pub fn uptime_secs(&self) -> i64 {
        (Utc::now() - self.started_at).num_seconds()
    }
}

// =============================================================================
// API CONFIG
// =============================================================================

/// Configuration for the API server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Host to bind to.
    pub host: String,
    /// Port to bind to.
    pub port: u16,
    /// CORS allowed origins.
    pub cors_origins: Vec<String>,
    /// Rate limit configuration.
    pub rate_limit: RateLimitConfig,
    /// Authentication configuration.
    pub auth: Option<AuthConfig>,
    /// Maximum request body size in bytes.
    pub max_body_size: usize,
    /// Request timeout in seconds.
    pub request_timeout_secs: u64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            cors_origins: vec!["*".to_string()],
            rate_limit: RateLimitConfig::default(),
            auth: None,
            max_body_size: 1024 * 1024, // 1MB
            request_timeout_secs: 30,
        }
    }
}

impl ApiConfig {
    /// Create a new configuration with the given host and port.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            ..Default::default()
        }
    }

    /// Get the bind address.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Rate limit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Whether rate limiting is enabled.
    pub enabled: bool,
    /// Maximum requests per window.
    pub max_requests: u32,
    /// Window size in seconds.
    pub window_secs: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_requests: 100,
            window_secs: 60,
        }
    }
}

/// Authentication configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Whether authentication is required.
    pub required: bool,
    /// Valid API keys.
    pub api_keys: Vec<String>,
    /// Paths that don't require authentication.
    pub public_paths: Vec<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            required: false,
            api_keys: Vec::new(),
            public_paths: vec!["/api/v1/health".to_string()],
        }
    }
}

// =============================================================================
// API SERVER
// =============================================================================

/// The main API server.
pub struct ApiServer {
    /// Server configuration.
    config: ApiConfig,
    /// HTTP router.
    router: Router,
    /// Shared state.
    state: Arc<ApiState>,
}

impl ApiServer {
    /// Create a new API server with the given configuration.
    pub fn new(config: ApiConfig) -> Self {
        Self {
            config,
            router: Router::new(),
            state: Arc::new(ApiState::new()),
        }
    }

    /// Create a builder for the API server.
    pub fn builder() -> ApiServerBuilder {
        ApiServerBuilder::default()
    }

    /// Get the server configuration.
    pub fn config(&self) -> &ApiConfig {
        &self.config
    }

    /// Get the shared state.
    pub fn state(&self) -> Arc<ApiState> {
        self.state.clone()
    }

    /// Get a reference to the router.
    pub fn router(&self) -> &Router {
        &self.router
    }

    /// Set the router.
    pub fn with_router(mut self, router: Router) -> Self {
        self.router = router;
        self
    }

    /// Build default routes for the API.
    pub fn with_default_routes(mut self) -> Self {
        self.router = Router::new()
            .get("/api/v1/health", handle_health)
            .get("/api/v1/metrics", handle_metrics)
            .get("/api/v1/agents", handle_list_agents)
            .get("/api/v1/agents/{agent_id}/status", handle_agent_status)
            .post("/api/v1/agents/{agent_id}/invoke", handle_invoke_agent)
            .post("/api/v1/query", handle_query);

        // Add default middleware
        self.router = self
            .router
            .middleware(CorsMiddleware::new(self.config.cors_origins.clone()));

        if self.config.rate_limit.enabled {
            self.router = self.router.middleware(RateLimitMiddleware::new(
                self.config.rate_limit.max_requests,
                self.config.rate_limit.window_secs,
            ));
        }

        if let Some(auth) = &self.config.auth {
            if auth.required {
                let mut auth_mw = AuthMiddleware::new(auth.api_keys.clone());
                for path in &auth.public_paths {
                    auth_mw = auth_mw.with_public_path(path);
                }
                self.router = self.router.middleware(auth_mw);
            }
        }

        self
    }

    /// Handle a single request.
    pub async fn handle_request(&self, request: Request) -> Response {
        let start = Instant::now();
        let response = self.router.handle(request, self.state.clone()).await;
        let duration_ms = start.elapsed().as_millis() as u64;

        self.state
            .record_request(response.status.is_success(), duration_ms);

        response
    }

    /// Generate OpenAPI documentation.
    pub fn openapi_spec(&self) -> serde_json::Value {
        let mut paths = serde_json::Map::new();

        for route in self.router.routes() {
            let method = route.method.to_string().to_lowercase();
            let operation = serde_json::json!({
                "summary": route.name.clone().unwrap_or_else(|| route.pattern.clone()),
                "description": route.description.clone().unwrap_or_default(),
                "responses": {
                    "200": { "description": "Success" },
                    "400": { "description": "Bad Request" },
                    "401": { "description": "Unauthorized" },
                    "404": { "description": "Not Found" },
                    "429": { "description": "Too Many Requests" },
                    "500": { "description": "Internal Server Error" }
                }
            });

            let path_entry = paths
                .entry(route.pattern.clone())
                .or_insert_with(|| serde_json::json!({}));
            if let serde_json::Value::Object(obj) = path_entry {
                obj.insert(method, operation);
            }
        }

        serde_json::json!({
            "openapi": "3.0.0",
            "info": {
                "title": "Panpsychism API",
                "version": "1.0.0",
                "description": "REST API for the Panpsychism prompt orchestration system"
            },
            "servers": [{
                "url": format!("http://{}", self.config.bind_addr()),
                "description": "Local server"
            }],
            "paths": paths
        })
    }
}

impl std::fmt::Debug for ApiServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiServer")
            .field("config", &self.config)
            .field("router", &self.router)
            .finish()
    }
}

// =============================================================================
// API SERVER BUILDER
// =============================================================================

/// Builder for creating an API server.
#[derive(Default)]
pub struct ApiServerBuilder {
    host: Option<String>,
    port: Option<u16>,
    cors_origins: Vec<String>,
    rate_limit_enabled: bool,
    rate_limit_max_requests: u32,
    rate_limit_window_secs: u64,
    auth_api_keys: Vec<String>,
    max_body_size: Option<usize>,
    request_timeout_secs: Option<u64>,
}

impl ApiServerBuilder {
    /// Set the host to bind to.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Set the port to bind to.
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Add a CORS allowed origin.
    pub fn cors_origin(mut self, origin: impl Into<String>) -> Self {
        self.cors_origins.push(origin.into());
        self
    }

    /// Set CORS allowed origins.
    pub fn cors_origins(mut self, origins: Vec<String>) -> Self {
        self.cors_origins = origins;
        self
    }

    /// Enable or disable rate limiting.
    pub fn rate_limit(mut self, enabled: bool) -> Self {
        self.rate_limit_enabled = enabled;
        self
    }

    /// Set rate limit configuration.
    pub fn rate_limit_config(mut self, max_requests: u32, window_secs: u64) -> Self {
        self.rate_limit_enabled = true;
        self.rate_limit_max_requests = max_requests;
        self.rate_limit_window_secs = window_secs;
        self
    }

    /// Add an API key for authentication.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.auth_api_keys.push(key.into());
        self
    }

    /// Set the maximum request body size.
    pub fn max_body_size(mut self, size: usize) -> Self {
        self.max_body_size = Some(size);
        self
    }

    /// Set the request timeout in seconds.
    pub fn request_timeout(mut self, secs: u64) -> Self {
        self.request_timeout_secs = Some(secs);
        self
    }

    /// Build the API server.
    pub fn build(self) -> Result<ApiServer> {
        let config = ApiConfig {
            host: self.host.unwrap_or_else(|| "127.0.0.1".to_string()),
            port: self.port.unwrap_or(8080),
            cors_origins: if self.cors_origins.is_empty() {
                vec!["*".to_string()]
            } else {
                self.cors_origins
            },
            rate_limit: RateLimitConfig {
                enabled: self.rate_limit_enabled,
                max_requests: if self.rate_limit_max_requests > 0 {
                    self.rate_limit_max_requests
                } else {
                    100
                },
                window_secs: if self.rate_limit_window_secs > 0 {
                    self.rate_limit_window_secs
                } else {
                    60
                },
            },
            auth: if self.auth_api_keys.is_empty() {
                None
            } else {
                Some(AuthConfig {
                    required: true,
                    api_keys: self.auth_api_keys,
                    public_paths: vec!["/api/v1/health".to_string()],
                })
            },
            max_body_size: self.max_body_size.unwrap_or(1024 * 1024),
            request_timeout_secs: self.request_timeout_secs.unwrap_or(30),
        };

        Ok(ApiServer::new(config).with_default_routes())
    }
}

// =============================================================================
// DEFAULT HANDLERS
// =============================================================================

/// Health check handler.
async fn handle_health(_request: Request, state: Arc<ApiState>) -> Response {
    let health = serde_json::json!({
        "status": "healthy",
        "uptime_secs": state.uptime_secs(),
        "timestamp": Utc::now().to_rfc3339(),
    });

    Response::ok()
        .with_json(&health)
        .unwrap_or_else(|_| Response::internal_error())
}

/// Metrics handler.
async fn handle_metrics(_request: Request, state: Arc<ApiState>) -> Response {
    let metrics = state.get_metrics();
    let response = serde_json::json!({
        "total_requests": metrics.total_requests,
        "successful_requests": metrics.successful_requests,
        "failed_requests": metrics.failed_requests,
        "success_rate": metrics.success_rate(),
        "avg_response_time_ms": metrics.avg_response_time_ms(),
        "uptime_secs": state.uptime_secs(),
    });

    Response::ok()
        .with_json(&response)
        .unwrap_or_else(|_| Response::internal_error())
}

/// List agents handler.
async fn handle_list_agents(_request: Request, state: Arc<ApiState>) -> Response {
    let agents = state.list_agents();
    let response = serde_json::json!({
        "agents": agents,
        "count": agents.len(),
    });

    Response::ok()
        .with_json(&response)
        .unwrap_or_else(|_| Response::internal_error())
}

/// Get agent status handler.
async fn handle_agent_status(request: Request, state: Arc<ApiState>) -> Response {
    let agent_id = match request.param("agent_id") {
        Some(id) => id,
        None => {
            return ApiError::bad_request("Missing agent_id parameter").into_response();
        }
    };

    match state.get_agent(agent_id) {
        Some(agent) => Response::ok()
            .with_json(&agent)
            .unwrap_or_else(|_| Response::internal_error()),
        None => ApiError::not_found(format!("Agent not found: {}", agent_id)).into_response(),
    }
}

/// Invoke agent handler.
async fn handle_invoke_agent(request: Request, state: Arc<ApiState>) -> Response {
    let agent_id = match request.param("agent_id") {
        Some(id) => id.to_string(),
        None => {
            return ApiError::bad_request("Missing agent_id parameter").into_response();
        }
    };

    // Check if agent exists
    if state.get_agent(&agent_id).is_none() {
        return ApiError::not_found(format!("Agent not found: {}", agent_id)).into_response();
    }

    // Parse request body
    let body: serde_json::Value = match request.body_json() {
        Ok(b) => b,
        Err(_) => {
            return ApiError::bad_request("Invalid JSON body").into_response();
        }
    };

    // Update agent status
    state.update_agent_status(&agent_id, AgentStatus::Busy);

    // Simulate processing (in real implementation, this would invoke the agent)
    let response = serde_json::json!({
        "agent_id": agent_id,
        "status": "invoked",
        "input": body,
        "timestamp": Utc::now().to_rfc3339(),
    });

    state.update_agent_status(&agent_id, AgentStatus::Idle);

    Response::ok()
        .with_json(&response)
        .unwrap_or_else(|_| Response::internal_error())
}

/// Query handler.
async fn handle_query(request: Request, _state: Arc<ApiState>) -> Response {
    // Parse request body
    let body: serde_json::Value = match request.body_json() {
        Ok(b) => b,
        Err(_) => {
            return ApiError::bad_request("Invalid JSON body").into_response();
        }
    };

    let query = body.get("query").and_then(|q| q.as_str()).unwrap_or("");

    if query.is_empty() {
        return ApiError::validation("Query cannot be empty").into_response();
    }

    // Simulate query processing (in real implementation, this would use the orchestrator)
    let response = serde_json::json!({
        "query": query,
        "status": "processed",
        "result": format!("Processed query: {}", query),
        "timestamp": Utc::now().to_rfc3339(),
    });

    Response::ok()
        .with_json(&response)
        .unwrap_or_else(|_| Response::internal_error())
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // METHOD TESTS
    // =========================================================================

    #[test]
    fn test_method_from_str() {
        assert_eq!(Method::from_str("GET"), Some(Method::GET));
        assert_eq!(Method::from_str("get"), Some(Method::GET));
        assert_eq!(Method::from_str("POST"), Some(Method::POST));
        assert_eq!(Method::from_str("put"), Some(Method::PUT));
        assert_eq!(Method::from_str("DELETE"), Some(Method::DELETE));
        assert_eq!(Method::from_str("patch"), Some(Method::PATCH));
        assert_eq!(Method::from_str("HEAD"), Some(Method::HEAD));
        assert_eq!(Method::from_str("OPTIONS"), Some(Method::OPTIONS));
        assert_eq!(Method::from_str("INVALID"), None);
    }

    #[test]
    fn test_method_display() {
        assert_eq!(format!("{}", Method::GET), "GET");
        assert_eq!(format!("{}", Method::POST), "POST");
        assert_eq!(format!("{}", Method::DELETE), "DELETE");
    }

    #[test]
    fn test_method_has_body() {
        assert!(!Method::GET.has_body());
        assert!(Method::POST.has_body());
        assert!(Method::PUT.has_body());
        assert!(Method::PATCH.has_body());
        assert!(!Method::DELETE.has_body());
        assert!(!Method::HEAD.has_body());
        assert!(!Method::OPTIONS.has_body());
    }

    #[test]
    fn test_method_is_safe() {
        assert!(Method::GET.is_safe());
        assert!(Method::HEAD.is_safe());
        assert!(Method::OPTIONS.is_safe());
        assert!(!Method::POST.is_safe());
        assert!(!Method::PUT.is_safe());
        assert!(!Method::DELETE.is_safe());
    }

    #[test]
    fn test_method_is_idempotent() {
        assert!(Method::GET.is_idempotent());
        assert!(Method::HEAD.is_idempotent());
        assert!(Method::PUT.is_idempotent());
        assert!(Method::DELETE.is_idempotent());
        assert!(!Method::POST.is_idempotent());
        assert!(!Method::PATCH.is_idempotent());
    }

    // =========================================================================
    // STATUS TESTS
    // =========================================================================

    #[test]
    fn test_status_constants() {
        assert_eq!(Status::OK.code(), 200);
        assert_eq!(Status::CREATED.code(), 201);
        assert_eq!(Status::BAD_REQUEST.code(), 400);
        assert_eq!(Status::NOT_FOUND.code(), 404);
        assert_eq!(Status::INTERNAL_SERVER_ERROR.code(), 500);
    }

    #[test]
    fn test_status_reason() {
        assert_eq!(Status::OK.reason(), "OK");
        assert_eq!(Status::NOT_FOUND.reason(), "Not Found");
        assert_eq!(Status::INTERNAL_SERVER_ERROR.reason(), "Internal Server Error");
        assert_eq!(Status(999).reason(), "Unknown");
    }

    #[test]
    fn test_status_is_success() {
        assert!(Status::OK.is_success());
        assert!(Status::CREATED.is_success());
        assert!(Status::NO_CONTENT.is_success());
        assert!(!Status::BAD_REQUEST.is_success());
        assert!(!Status::NOT_FOUND.is_success());
        assert!(!Status::INTERNAL_SERVER_ERROR.is_success());
    }

    #[test]
    fn test_status_is_client_error() {
        assert!(!Status::OK.is_client_error());
        assert!(Status::BAD_REQUEST.is_client_error());
        assert!(Status::NOT_FOUND.is_client_error());
        assert!(Status::TOO_MANY_REQUESTS.is_client_error());
        assert!(!Status::INTERNAL_SERVER_ERROR.is_client_error());
    }

    #[test]
    fn test_status_is_server_error() {
        assert!(!Status::OK.is_server_error());
        assert!(!Status::BAD_REQUEST.is_server_error());
        assert!(Status::INTERNAL_SERVER_ERROR.is_server_error());
        assert!(Status::SERVICE_UNAVAILABLE.is_server_error());
    }

    #[test]
    fn test_status_is_error() {
        assert!(!Status::OK.is_error());
        assert!(Status::BAD_REQUEST.is_error());
        assert!(Status::INTERNAL_SERVER_ERROR.is_error());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(format!("{}", Status::OK), "200 OK");
        assert_eq!(format!("{}", Status::NOT_FOUND), "404 Not Found");
    }

    // =========================================================================
    // REQUEST TESTS
    // =========================================================================

    #[test]
    fn test_request_new() {
        let req = Request::new(Method::GET, "/api/v1/test");
        assert_eq!(req.method, Method::GET);
        assert_eq!(req.path, "/api/v1/test");
        assert!(!req.id.is_empty());
    }

    #[test]
    fn test_request_with_header() {
        let req = Request::new(Method::GET, "/")
            .with_header("Content-Type", "application/json")
            .with_header("X-Custom", "value");

        assert_eq!(req.header("content-type"), Some("application/json"));
        assert_eq!(req.header("x-custom"), Some("value"));
        assert_eq!(req.header("missing"), None);
    }

    #[test]
    fn test_request_with_query() {
        let req = Request::new(Method::GET, "/")
            .with_query("page", "1")
            .with_query("limit", "10");

        assert_eq!(req.query("page"), Some("1"));
        assert_eq!(req.query("limit"), Some("10"));
        assert_eq!(req.query("missing"), None);
    }

    #[test]
    fn test_request_with_body() {
        let req = Request::new(Method::POST, "/").with_body(b"test body".to_vec());

        assert_eq!(req.body, b"test body".to_vec());
        assert_eq!(req.body_string().unwrap(), "test body");
    }

    #[test]
    fn test_request_with_json() {
        let data = serde_json::json!({"key": "value"});
        let req = Request::new(Method::POST, "/").with_json(&data).unwrap();

        assert_eq!(req.content_type(), Some("application/json"));
        let parsed: serde_json::Value = req.body_json().unwrap();
        assert_eq!(parsed["key"], "value");
    }

    #[test]
    fn test_request_bearer_token() {
        let req = Request::new(Method::GET, "/")
            .with_header("Authorization", "Bearer test-token-123");

        assert_eq!(req.bearer_token(), Some("test-token-123"));

        let req2 = Request::new(Method::GET, "/")
            .with_header("Authorization", "Basic dXNlcjpwYXNz");
        assert_eq!(req2.bearer_token(), None);

        let req3 = Request::new(Method::GET, "/");
        assert_eq!(req3.bearer_token(), None);
    }

    #[test]
    fn test_request_accepts_json() {
        let req1 = Request::new(Method::GET, "/")
            .with_header("Accept", "application/json");
        assert!(req1.accepts_json());

        let req2 = Request::new(Method::GET, "/")
            .with_header("Accept", "*/*");
        assert!(req2.accepts_json());

        let req3 = Request::new(Method::GET, "/");
        assert!(req3.accepts_json());
    }

    #[test]
    fn test_request_content_length() {
        let req = Request::new(Method::POST, "/")
            .with_header("Content-Length", "42");

        assert_eq!(req.content_length(), Some(42));
    }

    #[test]
    fn test_request_client_ip() {
        let req = Request::new(Method::GET, "/").with_client_ip("192.168.1.1");

        assert_eq!(req.client_ip, Some("192.168.1.1".to_string()));
    }

    // =========================================================================
    // RESPONSE TESTS
    // =========================================================================

    #[test]
    fn test_response_constructors() {
        assert_eq!(Response::ok().status.code(), 200);
        assert_eq!(Response::created().status.code(), 201);
        assert_eq!(Response::no_content().status.code(), 204);
        assert_eq!(Response::bad_request().status.code(), 400);
        assert_eq!(Response::unauthorized().status.code(), 401);
        assert_eq!(Response::forbidden().status.code(), 403);
        assert_eq!(Response::not_found().status.code(), 404);
        assert_eq!(Response::internal_error().status.code(), 500);
    }

    #[test]
    fn test_response_with_header() {
        let resp = Response::ok()
            .with_header("X-Custom", "value")
            .with_header("Content-Type", "text/plain");

        assert_eq!(resp.header("x-custom"), Some("value"));
        assert_eq!(resp.header("content-type"), Some("text/plain"));
    }

    #[test]
    fn test_response_with_body() {
        let resp = Response::ok().with_body(b"test".to_vec());

        assert_eq!(resp.body, b"test".to_vec());
    }

    #[test]
    fn test_response_with_json() {
        let data = serde_json::json!({"status": "ok"});
        let resp = Response::ok().with_json(&data).unwrap();

        assert_eq!(resp.header("content-type"), Some("application/json"));
        let parsed: serde_json::Value = resp.body_json().unwrap();
        assert_eq!(parsed["status"], "ok");
    }

    #[test]
    fn test_response_with_text() {
        let resp = Response::ok().with_text("Hello, World!");

        assert_eq!(resp.header("content-type"), Some("text/plain"));
        assert_eq!(resp.body_string().unwrap(), "Hello, World!");
    }

    #[test]
    fn test_response_is_success() {
        assert!(Response::ok().is_success());
        assert!(Response::created().is_success());
        assert!(!Response::bad_request().is_success());
        assert!(!Response::internal_error().is_success());
    }

    #[test]
    fn test_response_is_error() {
        assert!(!Response::ok().is_error());
        assert!(Response::bad_request().is_error());
        assert!(Response::not_found().is_error());
        assert!(Response::internal_error().is_error());
    }

    // =========================================================================
    // API ERROR TESTS
    // =========================================================================

    #[test]
    fn test_api_error_constructors() {
        let err = ApiError::bad_request("Invalid input");
        assert_eq!(err.code, "BAD_REQUEST");
        assert_eq!(err.status, 400);

        let err = ApiError::not_found("Resource not found");
        assert_eq!(err.code, "NOT_FOUND");
        assert_eq!(err.status, 404);

        let err = ApiError::unauthorized("Invalid token");
        assert_eq!(err.code, "UNAUTHORIZED");
        assert_eq!(err.status, 401);
    }

    #[test]
    fn test_api_error_with_details() {
        let err = ApiError::bad_request("Validation failed")
            .with_details(serde_json::json!({"field": "email"}));

        assert!(err.details.is_some());
        assert_eq!(err.details.unwrap()["field"], "email");
    }

    #[test]
    fn test_api_error_with_request_id() {
        let err = ApiError::internal("Something went wrong")
            .with_request_id("req-123");

        assert_eq!(err.request_id, Some("req-123".to_string()));
    }

    #[test]
    fn test_api_error_rate_limited() {
        let err = ApiError::rate_limited(Some(60));
        assert_eq!(err.code, "RATE_LIMITED");
        assert_eq!(err.status, 429);
        assert!(err.details.is_some());
    }

    #[test]
    fn test_api_error_into_response() {
        let err = ApiError::not_found("Agent not found");
        let resp = err.into_response();

        assert_eq!(resp.status.code(), 404);
        assert_eq!(resp.header("content-type"), Some("application/json"));
    }

    // =========================================================================
    // ROUTE TESTS
    // =========================================================================

    #[test]
    fn test_route_matches_literal() {
        let route = Route::new(Method::GET, "/api/v1/health", |_, _| async {
            Response::ok()
        });

        assert!(route.matches("/api/v1/health").is_some());
        assert!(route.matches("/api/v1/other").is_none());
        assert!(route.matches("/api/v1/health/extra").is_none());
    }

    #[test]
    fn test_route_matches_with_param() {
        let route = Route::new(Method::GET, "/api/v1/agents/{agent_id}", |_, _| async {
            Response::ok()
        });

        let params = route.matches("/api/v1/agents/agent-123").unwrap();
        assert_eq!(params.get("agent_id"), Some(&"agent-123".to_string()));

        let params = route.matches("/api/v1/agents/another").unwrap();
        assert_eq!(params.get("agent_id"), Some(&"another".to_string()));

        assert!(route.matches("/api/v1/agents").is_none());
        assert!(route.matches("/api/v1/agents/a/b").is_none());
    }

    #[test]
    fn test_route_matches_multiple_params() {
        let route = Route::new(Method::GET, "/api/v1/{version}/agents/{id}", |_, _| async {
            Response::ok()
        });

        let params = route.matches("/api/v1/beta/agents/123").unwrap();
        assert_eq!(params.get("version"), Some(&"beta".to_string()));
        assert_eq!(params.get("id"), Some(&"123".to_string()));
    }

    #[test]
    fn test_route_matches_wildcard() {
        let route = Route::new(Method::GET, "/static/*", |_, _| async {
            Response::ok()
        });

        assert!(route.matches("/static/file.js").is_some());
        assert!(route.matches("/static/dir/file.css").is_some());
        assert!(route.matches("/other").is_none());
    }

    #[test]
    fn test_route_with_name_and_description() {
        let route = Route::new(Method::GET, "/api/v1/health", |_, _| async {
            Response::ok()
        })
        .with_name("Health Check")
        .with_description("Returns server health status");

        assert_eq!(route.name, Some("Health Check".to_string()));
        assert_eq!(route.description, Some("Returns server health status".to_string()));
    }

    // =========================================================================
    // ROUTER TESTS
    // =========================================================================

    #[tokio::test]
    async fn test_router_find_route() {
        let router = Router::new()
            .get("/api/v1/health", |_, _| async { Response::ok() })
            .post("/api/v1/query", |_, _| async { Response::created() });

        let (route, _) = router.find_route(Method::GET, "/api/v1/health").unwrap();
        assert_eq!(route.method, Method::GET);

        let (route, _) = router.find_route(Method::POST, "/api/v1/query").unwrap();
        assert_eq!(route.method, Method::POST);

        assert!(router.find_route(Method::DELETE, "/api/v1/health").is_none());
        assert!(router.find_route(Method::GET, "/api/v1/other").is_none());
    }

    #[tokio::test]
    async fn test_router_path_has_routes() {
        let router = Router::new()
            .get("/api/v1/health", |_, _| async { Response::ok() })
            .post("/api/v1/health", |_, _| async { Response::created() });

        assert!(router.path_has_routes("/api/v1/health"));
        assert!(!router.path_has_routes("/api/v1/other"));
    }

    #[tokio::test]
    async fn test_router_handle_success() {
        let router = Router::new()
            .get("/api/v1/health", |_, _| async { Response::ok() });

        let state = Arc::new(ApiState::new());
        let request = Request::new(Method::GET, "/api/v1/health");

        let response = router.handle(request, state).await;
        assert_eq!(response.status.code(), 200);
    }

    #[tokio::test]
    async fn test_router_handle_not_found() {
        let router = Router::new()
            .get("/api/v1/health", |_, _| async { Response::ok() });

        let state = Arc::new(ApiState::new());
        let request = Request::new(Method::GET, "/api/v1/other");

        let response = router.handle(request, state).await;
        assert_eq!(response.status.code(), 404);
    }

    #[tokio::test]
    async fn test_router_handle_method_not_allowed() {
        let router = Router::new()
            .get("/api/v1/health", |_, _| async { Response::ok() });

        let state = Arc::new(ApiState::new());
        let request = Request::new(Method::DELETE, "/api/v1/health");

        let response = router.handle(request, state).await;
        assert_eq!(response.status.code(), 405);
    }

    #[tokio::test]
    async fn test_router_with_path_params() {
        let router = Router::new()
            .get("/api/v1/agents/{agent_id}", |req, _| async move {
                let id = req.param("agent_id").unwrap_or("unknown");
                Response::ok().with_text(format!("Agent: {}", id))
            });

        let state = Arc::new(ApiState::new());
        let request = Request::new(Method::GET, "/api/v1/agents/test-agent");

        let response = router.handle(request, state).await;
        assert_eq!(response.status.code(), 200);
        assert!(response.body_string().unwrap().contains("test-agent"));
    }

    // =========================================================================
    // MIDDLEWARE TESTS
    // =========================================================================

    #[test]
    fn test_cors_middleware_default() {
        let cors = CorsMiddleware::default();
        assert!(cors.is_origin_allowed("http://localhost:3000"));
        assert!(cors.is_origin_allowed("https://example.com"));
    }

    #[test]
    fn test_cors_middleware_specific_origins() {
        let cors = CorsMiddleware::new(vec!["http://localhost:3000".to_string()]);
        assert!(cors.is_origin_allowed("http://localhost:3000"));
        assert!(!cors.is_origin_allowed("https://example.com"));
    }

    #[test]
    fn test_cors_middleware_before_options() {
        let cors = CorsMiddleware::default();
        let state = ApiState::new();
        let request = Request::new(Method::OPTIONS, "/api/v1/health")
            .with_header("Origin", "http://localhost:3000");

        let response = cors.before(&request, &state);
        assert!(response.is_some());

        let resp = response.unwrap();
        assert_eq!(resp.status.code(), 204);
        assert!(resp.header("access-control-allow-origin").is_some());
    }

    #[test]
    fn test_cors_middleware_after() {
        let cors = CorsMiddleware::default();
        let state = ApiState::new();
        let request = Request::new(Method::GET, "/api/v1/health")
            .with_header("Origin", "http://localhost:3000");

        let response = Response::ok();
        let response = cors.after(&request, response, &state);

        assert!(response.header("access-control-allow-origin").is_some());
    }

    #[test]
    fn test_rate_limit_middleware_allows_within_limit() {
        let mw = RateLimitMiddleware::new(10, 60);
        let state = ApiState::new();
        let request = Request::new(Method::GET, "/api/v1/health")
            .with_client_ip("192.168.1.1");

        // First request should be allowed
        let response = mw.before(&request, &state);
        assert!(response.is_none());
    }

    #[test]
    fn test_rate_limit_middleware_blocks_over_limit() {
        let mw = RateLimitMiddleware::new(2, 60);
        let state = ApiState::new();
        let request = Request::new(Method::GET, "/api/v1/health")
            .with_client_ip("192.168.1.1");

        // First two requests should be allowed
        assert!(mw.before(&request, &state).is_none());
        assert!(mw.before(&request, &state).is_none());

        // Third request should be blocked
        let response = mw.before(&request, &state);
        assert!(response.is_some());
        assert_eq!(response.unwrap().status.code(), 429);
    }

    #[test]
    fn test_auth_middleware_public_path() {
        let mw = AuthMiddleware::new(vec!["secret-key".to_string()]);
        let state = ApiState::new();
        let request = Request::new(Method::GET, "/api/v1/health");

        // Health endpoint is public by default
        let response = mw.before(&request, &state);
        assert!(response.is_none());
    }

    #[test]
    fn test_auth_middleware_missing_token() {
        let mw = AuthMiddleware::new(vec!["secret-key".to_string()]);
        let state = ApiState::new();
        let request = Request::new(Method::GET, "/api/v1/agents");

        let response = mw.before(&request, &state);
        assert!(response.is_some());
        assert_eq!(response.unwrap().status.code(), 401);
    }

    #[test]
    fn test_auth_middleware_valid_token() {
        let mw = AuthMiddleware::new(vec!["secret-key".to_string()]);
        let state = ApiState::new();
        let request = Request::new(Method::GET, "/api/v1/agents")
            .with_header("Authorization", "Bearer secret-key");

        let response = mw.before(&request, &state);
        assert!(response.is_none());
    }

    #[test]
    fn test_auth_middleware_invalid_token() {
        let mw = AuthMiddleware::new(vec!["secret-key".to_string()]);
        let state = ApiState::new();
        let request = Request::new(Method::GET, "/api/v1/agents")
            .with_header("Authorization", "Bearer wrong-key");

        let response = mw.before(&request, &state);
        assert!(response.is_some());
        assert_eq!(response.unwrap().status.code(), 401);
    }

    // =========================================================================
    // API STATE TESTS
    // =========================================================================

    #[test]
    fn test_api_state_register_agent() {
        let state = ApiState::new();
        state.register_agent("agent-1", "Test Agent");

        let agent = state.get_agent("agent-1").unwrap();
        assert_eq!(agent.id, "agent-1");
        assert_eq!(agent.name, "Test Agent");
        assert_eq!(agent.status, AgentStatus::Idle);
    }

    #[test]
    fn test_api_state_list_agents() {
        let state = ApiState::new();
        state.register_agent("agent-1", "Agent One");
        state.register_agent("agent-2", "Agent Two");

        let agents = state.list_agents();
        assert_eq!(agents.len(), 2);
    }

    #[test]
    fn test_api_state_update_agent_status() {
        let state = ApiState::new();
        state.register_agent("agent-1", "Test Agent");

        assert!(state.update_agent_status("agent-1", AgentStatus::Busy));
        let agent = state.get_agent("agent-1").unwrap();
        assert_eq!(agent.status, AgentStatus::Busy);

        assert!(!state.update_agent_status("nonexistent", AgentStatus::Active));
    }

    #[test]
    fn test_api_state_record_request() {
        let state = ApiState::new();

        state.record_request(true, 100);
        state.record_request(true, 200);
        state.record_request(false, 50);

        let metrics = state.get_metrics();
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.successful_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
    }

    #[test]
    fn test_server_metrics_success_rate() {
        let mut metrics = ServerMetrics::default();
        metrics.record_request(true, 100);
        metrics.record_request(true, 100);
        metrics.record_request(false, 100);

        assert!((metrics.success_rate() - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_server_metrics_avg_response_time() {
        let mut metrics = ServerMetrics::default();
        metrics.record_request(true, 100);
        metrics.record_request(true, 200);
        metrics.record_request(true, 300);

        assert!((metrics.avg_response_time_ms() - 200.0).abs() < 0.1);
    }

    // =========================================================================
    // API CONFIG TESTS
    // =========================================================================

    #[test]
    fn test_api_config_default() {
        let config = ApiConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
        assert_eq!(config.cors_origins, vec!["*"]);
        assert!(config.rate_limit.enabled);
        assert!(config.auth.is_none());
    }

    #[test]
    fn test_api_config_bind_addr() {
        let config = ApiConfig::new("0.0.0.0", 9000);
        assert_eq!(config.bind_addr(), "0.0.0.0:9000");
    }

    // =========================================================================
    // API SERVER BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_api_server_builder() {
        let server = ApiServer::builder()
            .host("0.0.0.0")
            .port(9000)
            .cors_origin("http://localhost:3000")
            .rate_limit_config(50, 30)
            .api_key("test-key")
            .max_body_size(2 * 1024 * 1024)
            .request_timeout(60)
            .build()
            .unwrap();

        assert_eq!(server.config().host, "0.0.0.0");
        assert_eq!(server.config().port, 9000);
        assert_eq!(server.config().rate_limit.max_requests, 50);
        assert!(server.config().auth.is_some());
    }

    #[test]
    fn test_api_server_builder_defaults() {
        let server = ApiServer::builder().build().unwrap();

        assert_eq!(server.config().host, "127.0.0.1");
        assert_eq!(server.config().port, 8080);
    }

    // =========================================================================
    // API SERVER TESTS
    // =========================================================================

    #[tokio::test]
    async fn test_api_server_handle_health() {
        let server = ApiServer::builder().build().unwrap();
        let request = Request::new(Method::GET, "/api/v1/health");

        let response = server.handle_request(request).await;
        assert_eq!(response.status.code(), 200);

        let body: serde_json::Value = response.body_json().unwrap();
        assert_eq!(body["status"], "healthy");
    }

    #[tokio::test]
    async fn test_api_server_handle_metrics() {
        let server = ApiServer::builder().build().unwrap();

        // Make a few requests first
        server.handle_request(Request::new(Method::GET, "/api/v1/health")).await;
        server.handle_request(Request::new(Method::GET, "/api/v1/health")).await;

        let request = Request::new(Method::GET, "/api/v1/metrics");
        let response = server.handle_request(request).await;

        assert_eq!(response.status.code(), 200);
        let body: serde_json::Value = response.body_json().unwrap();
        assert!(body["total_requests"].as_u64().unwrap() >= 2);
    }

    #[tokio::test]
    async fn test_api_server_handle_agents_list() {
        let server = ApiServer::builder().build().unwrap();
        server.state().register_agent("agent-1", "Test Agent");

        let request = Request::new(Method::GET, "/api/v1/agents");
        let response = server.handle_request(request).await;

        assert_eq!(response.status.code(), 200);
        let body: serde_json::Value = response.body_json().unwrap();
        assert_eq!(body["count"], 1);
    }

    #[tokio::test]
    async fn test_api_server_handle_agent_status() {
        let server = ApiServer::builder().build().unwrap();
        server.state().register_agent("agent-1", "Test Agent");

        let request = Request::new(Method::GET, "/api/v1/agents/agent-1/status");
        let response = server.handle_request(request).await;

        assert_eq!(response.status.code(), 200);
        let body: serde_json::Value = response.body_json().unwrap();
        assert_eq!(body["id"], "agent-1");
    }

    #[tokio::test]
    async fn test_api_server_handle_agent_not_found() {
        let server = ApiServer::builder().build().unwrap();

        let request = Request::new(Method::GET, "/api/v1/agents/nonexistent/status");
        let response = server.handle_request(request).await;

        assert_eq!(response.status.code(), 404);
    }

    #[tokio::test]
    async fn test_api_server_handle_query() {
        let server = ApiServer::builder().build().unwrap();

        let body = serde_json::json!({"query": "test query"});
        let request = Request::new(Method::POST, "/api/v1/query")
            .with_json(&body)
            .unwrap();

        let response = server.handle_request(request).await;
        assert_eq!(response.status.code(), 200);

        let resp_body: serde_json::Value = response.body_json().unwrap();
        assert_eq!(resp_body["query"], "test query");
    }

    #[tokio::test]
    async fn test_api_server_handle_query_empty() {
        let server = ApiServer::builder().build().unwrap();

        let body = serde_json::json!({"query": ""});
        let request = Request::new(Method::POST, "/api/v1/query")
            .with_json(&body)
            .unwrap();

        let response = server.handle_request(request).await;
        assert_eq!(response.status.code(), 422);
    }

    #[tokio::test]
    async fn test_api_server_handle_invoke_agent() {
        let server = ApiServer::builder().build().unwrap();
        server.state().register_agent("agent-1", "Test Agent");

        let body = serde_json::json!({"input": "test"});
        let request = Request::new(Method::POST, "/api/v1/agents/agent-1/invoke")
            .with_json(&body)
            .unwrap();

        let response = server.handle_request(request).await;
        assert_eq!(response.status.code(), 200);

        let resp_body: serde_json::Value = response.body_json().unwrap();
        assert_eq!(resp_body["agent_id"], "agent-1");
        assert_eq!(resp_body["status"], "invoked");
    }

    #[tokio::test]
    async fn test_api_server_openapi_spec() {
        let server = ApiServer::builder().build().unwrap();
        let spec = server.openapi_spec();

        assert_eq!(spec["openapi"], "3.0.0");
        assert_eq!(spec["info"]["title"], "Panpsychism API");
        assert!(spec["paths"].as_object().is_some());
    }

    // =========================================================================
    // INTEGRATION TESTS
    // =========================================================================

    #[tokio::test]
    async fn test_full_request_flow_with_middleware() {
        let server = ApiServer::builder()
            .cors_origin("http://localhost:3000")
            .rate_limit_config(100, 60)
            .build()
            .unwrap();

        // Make request
        let request = Request::new(Method::GET, "/api/v1/health")
            .with_header("Origin", "http://localhost:3000");

        let response = server.handle_request(request).await;

        // Check response
        assert_eq!(response.status.code(), 200);
        assert!(response.header("access-control-allow-origin").is_some());
        assert!(response.header("x-ratelimit-limit").is_some());
    }

    #[tokio::test]
    async fn test_cors_preflight() {
        let server = ApiServer::builder()
            .cors_origin("http://localhost:3000")
            .build()
            .unwrap();

        let request = Request::new(Method::OPTIONS, "/api/v1/agents")
            .with_header("Origin", "http://localhost:3000")
            .with_header("Access-Control-Request-Method", "POST");

        let response = server.handle_request(request).await;

        assert_eq!(response.status.code(), 204);
        assert!(response.header("access-control-allow-methods").is_some());
    }

    #[tokio::test]
    async fn test_rate_limiting_integration() {
        let server = ApiServer::builder()
            .rate_limit_config(2, 60)
            .build()
            .unwrap();

        // First two requests should succeed
        for _ in 0..2 {
            let request = Request::new(Method::GET, "/api/v1/health")
                .with_client_ip("192.168.1.1");
            let response = server.handle_request(request).await;
            assert_eq!(response.status.code(), 200);
        }

        // Third request should be rate limited
        let request = Request::new(Method::GET, "/api/v1/health")
            .with_client_ip("192.168.1.1");
        let response = server.handle_request(request).await;
        assert_eq!(response.status.code(), 429);
    }

    #[tokio::test]
    async fn test_auth_integration() {
        let server = ApiServer::builder()
            .api_key("secret-key")
            .build()
            .unwrap();

        // Request without auth should fail for protected endpoint
        let request = Request::new(Method::GET, "/api/v1/agents");
        let response = server.handle_request(request).await;
        assert_eq!(response.status.code(), 401);

        // Request with valid auth should succeed
        let request = Request::new(Method::GET, "/api/v1/agents")
            .with_header("Authorization", "Bearer secret-key");
        let response = server.handle_request(request).await;
        assert_eq!(response.status.code(), 200);

        // Health endpoint should work without auth
        let request = Request::new(Method::GET, "/api/v1/health");
        let response = server.handle_request(request).await;
        assert_eq!(response.status.code(), 200);
    }
}
