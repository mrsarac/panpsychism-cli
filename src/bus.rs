//! # Agent Communication Bus (ACB)
//!
//! Real-time inter-agent communication system with pub/sub, request/response,
//! and event patterns for the Panpsychism orchestration framework.
//!
//! ## Features
//!
//! - **Point-to-Point Messaging**: Direct agent-to-agent communication
//! - **Pub/Sub**: Topic-based broadcast messaging
//! - **Event System**: System-wide event propagation
//! - **Heartbeat Monitoring**: Agent health tracking
//! - **Message History**: Ring buffer for debugging and auditing
//!
//! ## Example
//!
//! ```rust
//! use panpsychism::bus::{AgentBus, BusConfig, Message, EventType};
//!
//! let mut bus = AgentBus::new(BusConfig::default());
//!
//! // Register agents
//! bus.register_agent("agent-1", "Analyzer");
//! bus.register_agent("agent-2", "Synthesizer");
//!
//! // Subscribe to topics
//! bus.subscribe("agent-1", "analysis");
//! bus.subscribe("agent-2", "analysis");
//!
//! // Broadcast a message
//! bus.broadcast("agent-1", "analysis", serde_json::json!({"result": "success"}));
//!
//! // Poll messages
//! let messages = bus.poll_messages("agent-2");
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

// ============================================================================
// Type Aliases
// ============================================================================

/// Unique identifier for an agent in the bus system.
pub type AgentId = String;

/// Topic name for pub/sub messaging.
pub type Topic = String;

/// Unique identifier for messages.
pub type MessageId = String;

// ============================================================================
// Core Enums
// ============================================================================

/// Types of events that can occur in the system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// Agent has started and is ready to receive messages.
    AgentStarted,
    /// Agent has stopped and will no longer process messages.
    AgentStopped,
    /// Agent encountered an error.
    AgentError,
    /// A task was completed successfully.
    TaskCompleted,
    /// A task failed to complete.
    TaskFailed,
    /// Resource usage warning (memory, CPU, etc.).
    ResourceWarning,
    /// Configuration has changed.
    ConfigChanged,
    /// Custom event type for extensibility.
    Custom(String),
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::AgentStarted => write!(f, "AgentStarted"),
            EventType::AgentStopped => write!(f, "AgentStopped"),
            EventType::AgentError => write!(f, "AgentError"),
            EventType::TaskCompleted => write!(f, "TaskCompleted"),
            EventType::TaskFailed => write!(f, "TaskFailed"),
            EventType::ResourceWarning => write!(f, "ResourceWarning"),
            EventType::ConfigChanged => write!(f, "ConfigChanged"),
            EventType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Health status reported in heartbeat messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HeartbeatStatus {
    /// Agent is functioning normally.
    Healthy,
    /// Agent is experiencing some issues but still operational.
    Degraded,
    /// Agent is not functioning properly.
    Unhealthy,
}

impl Default for HeartbeatStatus {
    fn default() -> Self {
        HeartbeatStatus::Healthy
    }
}

/// Status of an agent in the bus system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is online and processing messages.
    Active,
    /// Agent is online but temporarily not processing.
    Idle,
    /// Agent is offline or unresponsive.
    Offline,
    /// Agent status is unknown (no recent heartbeat).
    Unknown,
}

impl Default for AgentStatus {
    fn default() -> Self {
        AgentStatus::Unknown
    }
}

// ============================================================================
// Message Types
// ============================================================================

/// Messages that can be sent through the agent bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// Direct request from one agent to another.
    Request {
        /// Unique message identifier.
        id: MessageId,
        /// Sender agent ID.
        from: AgentId,
        /// Target agent ID.
        to: AgentId,
        /// Message payload as JSON.
        payload: serde_json::Value,
        /// Timestamp when message was created.
        timestamp: DateTime<Utc>,
    },

    /// Response to a request message.
    Response {
        /// ID of the original request.
        id: MessageId,
        /// Target agent ID (original requester).
        to: AgentId,
        /// Response payload as JSON.
        payload: serde_json::Value,
        /// Whether the request was successful.
        success: bool,
        /// Timestamp when response was created.
        timestamp: DateTime<Utc>,
    },

    /// Broadcast message to all subscribers of a topic.
    Broadcast {
        /// Sender agent ID.
        from: AgentId,
        /// Topic to broadcast to.
        topic: Topic,
        /// Message payload as JSON.
        payload: serde_json::Value,
        /// Timestamp when message was created.
        timestamp: DateTime<Utc>,
    },

    /// System event notification.
    Event {
        /// Source agent ID.
        source: AgentId,
        /// Type of event.
        event_type: EventType,
        /// Event data as JSON.
        data: serde_json::Value,
        /// Timestamp when event occurred.
        timestamp: DateTime<Utc>,
    },

    /// Heartbeat message for health monitoring.
    Heartbeat {
        /// Agent sending the heartbeat.
        agent: AgentId,
        /// Current health status.
        status: HeartbeatStatus,
        /// Timestamp of heartbeat.
        timestamp: DateTime<Utc>,
    },
}

impl Message {
    /// Create a new request message.
    pub fn request(from: AgentId, to: AgentId, payload: serde_json::Value) -> Self {
        Message::Request {
            id: Uuid::new_v4().to_string(),
            from,
            to,
            payload,
            timestamp: Utc::now(),
        }
    }

    /// Create a new response message.
    pub fn response(request_id: MessageId, to: AgentId, payload: serde_json::Value, success: bool) -> Self {
        Message::Response {
            id: request_id,
            to,
            payload,
            success,
            timestamp: Utc::now(),
        }
    }

    /// Create a new broadcast message.
    pub fn broadcast(from: AgentId, topic: Topic, payload: serde_json::Value) -> Self {
        Message::Broadcast {
            from,
            topic,
            payload,
            timestamp: Utc::now(),
        }
    }

    /// Create a new event message.
    pub fn event(source: AgentId, event_type: EventType, data: serde_json::Value) -> Self {
        Message::Event {
            source,
            event_type,
            data,
            timestamp: Utc::now(),
        }
    }

    /// Create a new heartbeat message.
    pub fn heartbeat(agent: AgentId, status: HeartbeatStatus) -> Self {
        Message::Heartbeat {
            agent,
            status,
            timestamp: Utc::now(),
        }
    }

    /// Get the message ID if applicable.
    pub fn id(&self) -> Option<&str> {
        match self {
            Message::Request { id, .. } | Message::Response { id, .. } => Some(id),
            _ => None,
        }
    }

    /// Get the timestamp of the message.
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Message::Request { timestamp, .. }
            | Message::Response { timestamp, .. }
            | Message::Broadcast { timestamp, .. }
            | Message::Event { timestamp, .. }
            | Message::Heartbeat { timestamp, .. } => *timestamp,
        }
    }

    /// Get the source agent ID.
    pub fn source(&self) -> Option<&str> {
        match self {
            Message::Request { from, .. } => Some(from),
            Message::Broadcast { from, .. } => Some(from),
            Message::Event { source, .. } => Some(source),
            Message::Heartbeat { agent, .. } => Some(agent),
            Message::Response { .. } => None,
        }
    }

    /// Check if this is a heartbeat message.
    pub fn is_heartbeat(&self) -> bool {
        matches!(self, Message::Heartbeat { .. })
    }

    /// Check if this is an event message.
    pub fn is_event(&self) -> bool {
        matches!(self, Message::Event { .. })
    }
}

// ============================================================================
// Agent Info
// ============================================================================

/// Information about a registered agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Unique agent identifier.
    pub id: AgentId,
    /// Human-readable agent name.
    pub name: String,
    /// Current agent status.
    pub status: AgentStatus,
    /// Last heartbeat timestamp.
    pub last_heartbeat: Option<DateTime<Utc>>,
    /// Topics this agent is subscribed to.
    pub subscriptions: HashSet<Topic>,
    /// Timestamp when agent was registered.
    pub registered_at: DateTime<Utc>,
}

impl AgentInfo {
    /// Create a new agent info.
    pub fn new(id: AgentId, name: impl Into<String>) -> Self {
        AgentInfo {
            id,
            name: name.into(),
            status: AgentStatus::Active,
            last_heartbeat: None,
            subscriptions: HashSet::new(),
            registered_at: Utc::now(),
        }
    }

    /// Update the heartbeat timestamp and status.
    pub fn update_heartbeat(&mut self, status: HeartbeatStatus) {
        self.last_heartbeat = Some(Utc::now());
        self.status = match status {
            HeartbeatStatus::Healthy => AgentStatus::Active,
            HeartbeatStatus::Degraded => AgentStatus::Active,
            HeartbeatStatus::Unhealthy => AgentStatus::Offline,
        };
    }

    /// Check if the agent is considered healthy based on heartbeat.
    pub fn is_healthy(&self, timeout_ms: u64) -> bool {
        match self.last_heartbeat {
            Some(last) => {
                let elapsed = Utc::now().signed_duration_since(last);
                elapsed.num_milliseconds() < timeout_ms as i64
            }
            None => false,
        }
    }
}

// ============================================================================
// Bus Configuration
// ============================================================================

/// Configuration for the agent bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusConfig {
    /// Maximum number of messages in the queue per agent.
    pub max_queue_size: usize,
    /// Maximum number of messages to keep in the log (ring buffer).
    pub max_log_size: usize,
    /// Expected heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Timeout for considering an agent unhealthy.
    pub timeout_ms: u64,
    /// Whether to log all messages.
    pub enable_logging: bool,
    /// Maximum payload size in bytes.
    pub max_payload_size: usize,
}

impl Default for BusConfig {
    fn default() -> Self {
        BusConfig {
            max_queue_size: 1000,
            max_log_size: 10000,
            heartbeat_interval_ms: 5000,
            timeout_ms: 15000,
            enable_logging: true,
            max_payload_size: 1024 * 1024, // 1MB
        }
    }
}

impl BusConfig {
    /// Create a new configuration with custom values.
    pub fn new(
        max_queue_size: usize,
        max_log_size: usize,
        heartbeat_interval_ms: u64,
        timeout_ms: u64,
    ) -> Self {
        BusConfig {
            max_queue_size,
            max_log_size,
            heartbeat_interval_ms,
            timeout_ms,
            ..Default::default()
        }
    }

    /// Create a minimal configuration for testing.
    pub fn minimal() -> Self {
        BusConfig {
            max_queue_size: 100,
            max_log_size: 100,
            heartbeat_interval_ms: 1000,
            timeout_ms: 3000,
            enable_logging: false,
            max_payload_size: 10240,
        }
    }
}

// ============================================================================
// Bus Participant Trait
// ============================================================================

/// Trait for agents that can participate in the bus.
pub trait BusParticipant {
    /// Get the unique agent identifier.
    fn agent_id(&self) -> AgentId;

    /// Handle an incoming message and optionally return a response.
    fn on_message(&mut self, msg: &Message) -> Option<Message>;

    /// Get the agent's current health status.
    fn health_status(&self) -> HeartbeatStatus {
        HeartbeatStatus::Healthy
    }

    /// Get topics this agent wants to subscribe to.
    fn subscribed_topics(&self) -> Vec<Topic> {
        Vec::new()
    }
}

// ============================================================================
// Agent Bus
// ============================================================================

/// The central message bus for agent communication.
#[derive(Debug)]
pub struct AgentBus {
    /// Registered agents.
    agents: HashMap<AgentId, AgentInfo>,
    /// Topic subscriptions: topic -> set of agent IDs.
    subscriptions: HashMap<Topic, HashSet<AgentId>>,
    /// Message queues per agent: agent_id -> messages.
    message_queues: HashMap<AgentId, VecDeque<Message>>,
    /// Message history log (ring buffer).
    message_log: VecDeque<Message>,
    /// Bus configuration.
    config: BusConfig,
    /// Counter for statistics.
    stats: BusStats,
}

/// Statistics about bus activity.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BusStats {
    /// Total messages sent.
    pub messages_sent: u64,
    /// Total messages delivered.
    pub messages_delivered: u64,
    /// Total messages dropped (queue full).
    pub messages_dropped: u64,
    /// Total events published.
    pub events_published: u64,
    /// Total broadcasts sent.
    pub broadcasts_sent: u64,
    /// Total heartbeats received.
    pub heartbeats_received: u64,
}

impl AgentBus {
    /// Create a new agent bus with the given configuration.
    pub fn new(config: BusConfig) -> Self {
        AgentBus {
            agents: HashMap::new(),
            subscriptions: HashMap::new(),
            message_queues: HashMap::new(),
            message_log: VecDeque::with_capacity(config.max_log_size),
            config,
            stats: BusStats::default(),
        }
    }

    /// Create a new agent bus with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BusConfig::default())
    }

    /// Get the bus configuration.
    pub fn config(&self) -> &BusConfig {
        &self.config
    }

    /// Get bus statistics.
    pub fn stats(&self) -> &BusStats {
        &self.stats
    }

    // ========================================================================
    // Agent Management
    // ========================================================================

    /// Register a new agent with the bus.
    pub fn register_agent(&mut self, id: impl Into<AgentId>, name: impl Into<String>) -> bool {
        let id = id.into();
        if self.agents.contains_key(&id) {
            return false;
        }

        let info = AgentInfo::new(id.clone(), name);
        self.agents.insert(id.clone(), info);
        self.message_queues.insert(id, VecDeque::new());
        true
    }

    /// Unregister an agent from the bus.
    pub fn unregister_agent(&mut self, id: &str) -> bool {
        if self.agents.remove(id).is_none() {
            return false;
        }

        // Remove from all subscriptions
        for subscribers in self.subscriptions.values_mut() {
            subscribers.remove(id);
        }

        // Remove message queue
        self.message_queues.remove(id);
        true
    }

    /// Get information about a registered agent.
    pub fn get_agent(&self, id: &str) -> Option<&AgentInfo> {
        self.agents.get(id)
    }

    /// Get all registered agents.
    pub fn get_agents(&self) -> Vec<&AgentInfo> {
        self.agents.values().collect()
    }

    /// Get the number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Check if an agent is registered.
    pub fn is_registered(&self, id: &str) -> bool {
        self.agents.contains_key(id)
    }

    // ========================================================================
    // Subscription Management
    // ========================================================================

    /// Subscribe an agent to a topic.
    pub fn subscribe(&mut self, agent_id: &str, topic: impl Into<Topic>) -> bool {
        let topic = topic.into();

        // Check if agent exists
        let agent = match self.agents.get_mut(agent_id) {
            Some(a) => a,
            None => return false,
        };

        // Add to agent's subscriptions
        agent.subscriptions.insert(topic.clone());

        // Add to topic's subscriber list
        self.subscriptions
            .entry(topic)
            .or_insert_with(HashSet::new)
            .insert(agent_id.to_string());

        true
    }

    /// Unsubscribe an agent from a topic.
    pub fn unsubscribe(&mut self, agent_id: &str, topic: &str) -> bool {
        // Remove from agent's subscriptions
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.subscriptions.remove(topic);
        } else {
            return false;
        }

        // Remove from topic's subscriber list
        if let Some(subscribers) = self.subscriptions.get_mut(topic) {
            subscribers.remove(agent_id);
            if subscribers.is_empty() {
                self.subscriptions.remove(topic);
            }
        }

        true
    }

    /// Get all subscribers for a topic.
    pub fn get_subscribers(&self, topic: &str) -> Vec<&str> {
        self.subscriptions
            .get(topic)
            .map(|s| s.iter().map(|id| id.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all topics an agent is subscribed to.
    pub fn get_agent_subscriptions(&self, agent_id: &str) -> Vec<&str> {
        self.agents
            .get(agent_id)
            .map(|a| a.subscriptions.iter().map(|t| t.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all active topics.
    pub fn get_topics(&self) -> Vec<&str> {
        self.subscriptions.keys().map(|t| t.as_str()).collect()
    }

    // ========================================================================
    // Messaging
    // ========================================================================

    /// Send a point-to-point message from one agent to another.
    pub fn send(
        &mut self,
        from: impl Into<AgentId>,
        to: impl Into<AgentId>,
        payload: serde_json::Value,
    ) -> Option<MessageId> {
        let from = from.into();
        let to = to.into();

        // Verify both agents exist
        if !self.agents.contains_key(&from) || !self.agents.contains_key(&to) {
            return None;
        }

        let msg = Message::request(from, to.clone(), payload);
        let msg_id = msg.id().unwrap().to_string();

        self.enqueue_message(&to, msg);
        self.stats.messages_sent += 1;

        Some(msg_id)
    }

    /// Send a response to a previous request.
    pub fn respond(
        &mut self,
        request_id: impl Into<MessageId>,
        to: impl Into<AgentId>,
        payload: serde_json::Value,
        success: bool,
    ) -> bool {
        let to = to.into();

        if !self.agents.contains_key(&to) {
            return false;
        }

        let msg = Message::response(request_id.into(), to.clone(), payload, success);
        self.enqueue_message(&to, msg);
        self.stats.messages_sent += 1;

        true
    }

    /// Broadcast a message to all subscribers of a topic.
    pub fn broadcast(
        &mut self,
        from: impl Into<AgentId>,
        topic: impl Into<Topic>,
        payload: serde_json::Value,
    ) -> usize {
        let from = from.into();
        let topic = topic.into();

        // Verify sender exists
        if !self.agents.contains_key(&from) {
            return 0;
        }

        let subscribers: Vec<AgentId> = self
            .subscriptions
            .get(&topic)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();

        if subscribers.is_empty() {
            return 0;
        }

        let msg = Message::broadcast(from.clone(), topic, payload);
        let mut delivered = 0;

        for subscriber in &subscribers {
            if subscriber != &from {
                self.enqueue_message(subscriber, msg.clone());
                delivered += 1;
            }
        }

        self.stats.broadcasts_sent += 1;
        self.stats.messages_sent += delivered as u64;

        delivered
    }

    /// Publish an event to the system.
    pub fn publish_event(
        &mut self,
        source: impl Into<AgentId>,
        event_type: EventType,
        data: serde_json::Value,
    ) -> bool {
        let source = source.into();

        if !self.agents.contains_key(&source) {
            return false;
        }

        let msg = Message::event(source, event_type, data);

        // Events go to all agents
        let agent_ids: Vec<AgentId> = self.agents.keys().cloned().collect();
        for agent_id in agent_ids {
            self.enqueue_message(&agent_id, msg.clone());
        }

        self.stats.events_published += 1;
        true
    }

    /// Poll messages for a specific agent.
    pub fn poll_messages(&mut self, agent_id: &str) -> Vec<Message> {
        self.message_queues
            .get_mut(agent_id)
            .map(|queue| {
                let messages: Vec<Message> = queue.drain(..).collect();
                self.stats.messages_delivered += messages.len() as u64;
                messages
            })
            .unwrap_or_default()
    }

    /// Peek at messages without removing them.
    pub fn peek_messages(&self, agent_id: &str) -> Vec<&Message> {
        self.message_queues
            .get(agent_id)
            .map(|queue| queue.iter().collect())
            .unwrap_or_default()
    }

    /// Get the number of pending messages for an agent.
    pub fn pending_message_count(&self, agent_id: &str) -> usize {
        self.message_queues
            .get(agent_id)
            .map(|q| q.len())
            .unwrap_or(0)
    }

    // ========================================================================
    // Health Monitoring
    // ========================================================================

    /// Send a heartbeat for an agent.
    pub fn heartbeat(&mut self, agent_id: &str, status: HeartbeatStatus) -> bool {
        let agent = match self.agents.get_mut(agent_id) {
            Some(a) => a,
            None => return false,
        };

        agent.update_heartbeat(status);
        self.stats.heartbeats_received += 1;

        // Log heartbeat if enabled
        if self.config.enable_logging {
            let msg = Message::heartbeat(agent_id.to_string(), status);
            self.log_message(msg);
        }

        true
    }

    /// Check health of all agents.
    pub fn check_health(&mut self) -> HashMap<AgentId, (AgentStatus, bool)> {
        let timeout = self.config.timeout_ms;
        let mut health_report = HashMap::new();

        for (id, agent) in &mut self.agents {
            let is_healthy = agent.is_healthy(timeout);
            if !is_healthy && agent.status == AgentStatus::Active {
                agent.status = AgentStatus::Unknown;
            }
            health_report.insert(id.clone(), (agent.status, is_healthy));
        }

        health_report
    }

    /// Get unhealthy agents.
    pub fn get_unhealthy_agents(&self) -> Vec<&AgentInfo> {
        let timeout = self.config.timeout_ms;
        self.agents
            .values()
            .filter(|a| !a.is_healthy(timeout))
            .collect()
    }

    // ========================================================================
    // Message Log
    // ========================================================================

    /// Get recent messages from the log.
    pub fn get_message_log(&self, limit: usize) -> Vec<&Message> {
        self.message_log.iter().rev().take(limit).collect()
    }

    /// Clear the message log.
    pub fn clear_message_log(&mut self) {
        self.message_log.clear();
    }

    /// Get the size of the message log.
    pub fn message_log_size(&self) -> usize {
        self.message_log.len()
    }

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// Enqueue a message for an agent.
    fn enqueue_message(&mut self, agent_id: &str, msg: Message) {
        // Log message first
        if self.config.enable_logging {
            self.log_message(msg.clone());
        }

        // Get or create queue
        let queue = match self.message_queues.get_mut(agent_id) {
            Some(q) => q,
            None => return,
        };

        // Check queue size
        if queue.len() >= self.config.max_queue_size {
            // Drop oldest message
            queue.pop_front();
            self.stats.messages_dropped += 1;
        }

        queue.push_back(msg);
    }

    /// Log a message to the ring buffer.
    fn log_message(&mut self, msg: Message) {
        if self.message_log.len() >= self.config.max_log_size {
            self.message_log.pop_front();
        }
        self.message_log.push_back(msg);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ------------------------------------------------------------------------
    // Configuration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_bus_config_default() {
        let config = BusConfig::default();
        assert_eq!(config.max_queue_size, 1000);
        assert_eq!(config.max_log_size, 10000);
        assert_eq!(config.heartbeat_interval_ms, 5000);
        assert_eq!(config.timeout_ms, 15000);
        assert!(config.enable_logging);
    }

    #[test]
    fn test_bus_config_custom() {
        let config = BusConfig::new(500, 5000, 2000, 6000);
        assert_eq!(config.max_queue_size, 500);
        assert_eq!(config.max_log_size, 5000);
        assert_eq!(config.heartbeat_interval_ms, 2000);
        assert_eq!(config.timeout_ms, 6000);
    }

    #[test]
    fn test_bus_config_minimal() {
        let config = BusConfig::minimal();
        assert_eq!(config.max_queue_size, 100);
        assert_eq!(config.max_log_size, 100);
        assert!(!config.enable_logging);
    }

    // ------------------------------------------------------------------------
    // Agent Registration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_register_agent() {
        let mut bus = AgentBus::new(BusConfig::minimal());

        assert!(bus.register_agent("agent-1", "Test Agent"));
        assert!(bus.is_registered("agent-1"));
        assert_eq!(bus.agent_count(), 1);
    }

    #[test]
    fn test_register_agent_duplicate() {
        let mut bus = AgentBus::new(BusConfig::minimal());

        assert!(bus.register_agent("agent-1", "Test Agent"));
        assert!(!bus.register_agent("agent-1", "Test Agent 2"));
        assert_eq!(bus.agent_count(), 1);
    }

    #[test]
    fn test_unregister_agent() {
        let mut bus = AgentBus::new(BusConfig::minimal());

        bus.register_agent("agent-1", "Test Agent");
        assert!(bus.unregister_agent("agent-1"));
        assert!(!bus.is_registered("agent-1"));
        assert_eq!(bus.agent_count(), 0);
    }

    #[test]
    fn test_unregister_nonexistent_agent() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        assert!(!bus.unregister_agent("nonexistent"));
    }

    #[test]
    fn test_get_agent() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Test Agent");

        let agent = bus.get_agent("agent-1").unwrap();
        assert_eq!(agent.id, "agent-1");
        assert_eq!(agent.name, "Test Agent");
        assert_eq!(agent.status, AgentStatus::Active);
    }

    #[test]
    fn test_get_agents() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Agent One");
        bus.register_agent("agent-2", "Agent Two");

        let agents = bus.get_agents();
        assert_eq!(agents.len(), 2);
    }

    // ------------------------------------------------------------------------
    // Subscription Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_subscribe() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Test Agent");

        assert!(bus.subscribe("agent-1", "topic-1"));

        let subs = bus.get_subscribers("topic-1");
        assert_eq!(subs.len(), 1);
        assert!(subs.contains(&"agent-1"));
    }

    #[test]
    fn test_subscribe_nonexistent_agent() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        assert!(!bus.subscribe("nonexistent", "topic-1"));
    }

    #[test]
    fn test_subscribe_multiple_agents() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Agent One");
        bus.register_agent("agent-2", "Agent Two");

        bus.subscribe("agent-1", "topic-1");
        bus.subscribe("agent-2", "topic-1");

        let subs = bus.get_subscribers("topic-1");
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_unsubscribe() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Test Agent");
        bus.subscribe("agent-1", "topic-1");

        assert!(bus.unsubscribe("agent-1", "topic-1"));

        let subs = bus.get_subscribers("topic-1");
        assert!(subs.is_empty());
    }

    #[test]
    fn test_unsubscribe_nonexistent_agent() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        assert!(!bus.unsubscribe("nonexistent", "topic-1"));
    }

    #[test]
    fn test_get_agent_subscriptions() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Test Agent");
        bus.subscribe("agent-1", "topic-1");
        bus.subscribe("agent-1", "topic-2");

        let subs = bus.get_agent_subscriptions("agent-1");
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_get_topics() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Test Agent");
        bus.subscribe("agent-1", "topic-1");
        bus.subscribe("agent-1", "topic-2");

        let topics = bus.get_topics();
        assert_eq!(topics.len(), 2);
    }

    #[test]
    fn test_unregister_removes_subscriptions() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Test Agent");
        bus.subscribe("agent-1", "topic-1");

        bus.unregister_agent("agent-1");

        let subs = bus.get_subscribers("topic-1");
        assert!(subs.is_empty());
    }

    // ------------------------------------------------------------------------
    // Point-to-Point Messaging Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_send_message() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        let msg_id = bus.send("agent-1", "agent-2", json!({"hello": "world"}));
        assert!(msg_id.is_some());
    }

    #[test]
    fn test_send_message_nonexistent_sender() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-2", "Receiver");

        let msg_id = bus.send("nonexistent", "agent-2", json!({}));
        assert!(msg_id.is_none());
    }

    #[test]
    fn test_send_message_nonexistent_receiver() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Sender");

        let msg_id = bus.send("agent-1", "nonexistent", json!({}));
        assert!(msg_id.is_none());
    }

    #[test]
    fn test_poll_messages() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        bus.send("agent-1", "agent-2", json!({"data": 1}));
        bus.send("agent-1", "agent-2", json!({"data": 2}));

        let messages = bus.poll_messages("agent-2");
        assert_eq!(messages.len(), 2);

        // Queue should be empty after poll
        let messages = bus.poll_messages("agent-2");
        assert!(messages.is_empty());
    }

    #[test]
    fn test_peek_messages() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        bus.send("agent-1", "agent-2", json!({"data": 1}));

        // Peek should not remove messages
        let messages = bus.peek_messages("agent-2");
        assert_eq!(messages.len(), 1);

        let messages = bus.peek_messages("agent-2");
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn test_pending_message_count() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        assert_eq!(bus.pending_message_count("agent-2"), 0);

        bus.send("agent-1", "agent-2", json!({}));
        bus.send("agent-1", "agent-2", json!({}));

        assert_eq!(bus.pending_message_count("agent-2"), 2);
    }

    #[test]
    fn test_respond_to_request() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Requester");
        bus.register_agent("agent-2", "Responder");

        let msg_id = bus.send("agent-1", "agent-2", json!({"request": "data"})).unwrap();

        assert!(bus.respond(&msg_id, "agent-1", json!({"response": "ok"}), true));

        let messages = bus.poll_messages("agent-1");
        assert_eq!(messages.len(), 1);

        match &messages[0] {
            Message::Response { success, .. } => assert!(success),
            _ => panic!("Expected Response message"),
        }
    }

    // ------------------------------------------------------------------------
    // Broadcast Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_broadcast() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Broadcaster");
        bus.register_agent("agent-2", "Subscriber 1");
        bus.register_agent("agent-3", "Subscriber 2");

        bus.subscribe("agent-2", "news");
        bus.subscribe("agent-3", "news");

        let delivered = bus.broadcast("agent-1", "news", json!({"headline": "Breaking"}));
        assert_eq!(delivered, 2);

        assert_eq!(bus.pending_message_count("agent-2"), 1);
        assert_eq!(bus.pending_message_count("agent-3"), 1);
    }

    #[test]
    fn test_broadcast_excludes_sender() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Broadcaster");
        bus.register_agent("agent-2", "Subscriber");

        bus.subscribe("agent-1", "news");
        bus.subscribe("agent-2", "news");

        let delivered = bus.broadcast("agent-1", "news", json!({}));
        assert_eq!(delivered, 1);

        // Sender should not receive their own broadcast
        assert_eq!(bus.pending_message_count("agent-1"), 0);
    }

    #[test]
    fn test_broadcast_no_subscribers() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Broadcaster");

        let delivered = bus.broadcast("agent-1", "empty-topic", json!({}));
        assert_eq!(delivered, 0);
    }

    #[test]
    fn test_broadcast_nonexistent_sender() {
        let mut bus = AgentBus::new(BusConfig::minimal());

        let delivered = bus.broadcast("nonexistent", "topic", json!({}));
        assert_eq!(delivered, 0);
    }

    // ------------------------------------------------------------------------
    // Event Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_publish_event() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Source");
        bus.register_agent("agent-2", "Listener");

        assert!(bus.publish_event("agent-1", EventType::AgentStarted, json!({})));

        let messages = bus.poll_messages("agent-2");
        assert_eq!(messages.len(), 1);

        match &messages[0] {
            Message::Event { event_type, .. } => {
                assert_eq!(*event_type, EventType::AgentStarted);
            }
            _ => panic!("Expected Event message"),
        }
    }

    #[test]
    fn test_publish_event_nonexistent_source() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        assert!(!bus.publish_event("nonexistent", EventType::AgentStarted, json!({})));
    }

    #[test]
    fn test_event_reaches_all_agents() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Source");
        bus.register_agent("agent-2", "Listener 1");
        bus.register_agent("agent-3", "Listener 2");

        bus.publish_event("agent-1", EventType::TaskCompleted, json!({}));

        // All agents should receive the event
        assert!(bus.pending_message_count("agent-1") >= 1);
        assert!(bus.pending_message_count("agent-2") >= 1);
        assert!(bus.pending_message_count("agent-3") >= 1);
    }

    #[test]
    fn test_custom_event_type() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Source");

        let custom_event = EventType::Custom("MyEvent".to_string());
        assert!(bus.publish_event("agent-1", custom_event.clone(), json!({})));

        let messages = bus.poll_messages("agent-1");
        match &messages[0] {
            Message::Event { event_type, .. } => {
                assert_eq!(*event_type, EventType::Custom("MyEvent".to_string()));
            }
            _ => panic!("Expected Event message"),
        }
    }

    // ------------------------------------------------------------------------
    // Heartbeat Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_heartbeat() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Test Agent");

        assert!(bus.heartbeat("agent-1", HeartbeatStatus::Healthy));

        let agent = bus.get_agent("agent-1").unwrap();
        assert!(agent.last_heartbeat.is_some());
        assert_eq!(agent.status, AgentStatus::Active);
    }

    #[test]
    fn test_heartbeat_nonexistent_agent() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        assert!(!bus.heartbeat("nonexistent", HeartbeatStatus::Healthy));
    }

    #[test]
    fn test_heartbeat_degraded() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Test Agent");

        bus.heartbeat("agent-1", HeartbeatStatus::Degraded);

        let agent = bus.get_agent("agent-1").unwrap();
        assert_eq!(agent.status, AgentStatus::Active);
    }

    #[test]
    fn test_heartbeat_unhealthy() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Test Agent");

        bus.heartbeat("agent-1", HeartbeatStatus::Unhealthy);

        let agent = bus.get_agent("agent-1").unwrap();
        assert_eq!(agent.status, AgentStatus::Offline);
    }

    #[test]
    fn test_check_health() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Agent One");
        bus.register_agent("agent-2", "Agent Two");

        bus.heartbeat("agent-1", HeartbeatStatus::Healthy);
        // agent-2 has no heartbeat

        let health = bus.check_health();
        assert_eq!(health.len(), 2);

        let (_, is_healthy_1) = health.get("agent-1").unwrap();
        assert!(is_healthy_1);

        let (_, is_healthy_2) = health.get("agent-2").unwrap();
        assert!(!is_healthy_2);
    }

    #[test]
    fn test_get_unhealthy_agents() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Healthy Agent");
        bus.register_agent("agent-2", "Unhealthy Agent");

        bus.heartbeat("agent-1", HeartbeatStatus::Healthy);

        let unhealthy = bus.get_unhealthy_agents();
        assert_eq!(unhealthy.len(), 1);
        assert_eq!(unhealthy[0].id, "agent-2");
    }

    // ------------------------------------------------------------------------
    // Message Log Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_message_log() {
        let config = BusConfig {
            enable_logging: true,
            max_log_size: 10,
            ..BusConfig::minimal()
        };
        let mut bus = AgentBus::new(config);
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        bus.send("agent-1", "agent-2", json!({}));

        assert_eq!(bus.message_log_size(), 1);

        let log = bus.get_message_log(10);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_message_log_ring_buffer() {
        let config = BusConfig {
            enable_logging: true,
            max_log_size: 3,
            ..BusConfig::minimal()
        };
        let mut bus = AgentBus::new(config);
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        for i in 0..5 {
            bus.send("agent-1", "agent-2", json!({"index": i}));
        }

        // Ring buffer should only keep last 3
        assert_eq!(bus.message_log_size(), 3);
    }

    #[test]
    fn test_clear_message_log() {
        let mut bus = AgentBus::new(BusConfig::default());
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        bus.send("agent-1", "agent-2", json!({}));
        assert!(bus.message_log_size() > 0);

        bus.clear_message_log();
        assert_eq!(bus.message_log_size(), 0);
    }

    // ------------------------------------------------------------------------
    // Queue Management Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_queue_overflow() {
        let config = BusConfig {
            max_queue_size: 3,
            ..BusConfig::minimal()
        };
        let mut bus = AgentBus::new(config);
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        for i in 0..5 {
            bus.send("agent-1", "agent-2", json!({"index": i}));
        }

        // Queue should drop oldest messages
        let messages = bus.poll_messages("agent-2");
        assert_eq!(messages.len(), 3);

        // Check stats
        assert_eq!(bus.stats().messages_dropped, 2);
    }

    // ------------------------------------------------------------------------
    // Statistics Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_stats_messages_sent() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        bus.send("agent-1", "agent-2", json!({}));
        bus.send("agent-1", "agent-2", json!({}));

        assert_eq!(bus.stats().messages_sent, 2);
    }

    #[test]
    fn test_stats_messages_delivered() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Sender");
        bus.register_agent("agent-2", "Receiver");

        bus.send("agent-1", "agent-2", json!({}));
        bus.poll_messages("agent-2");

        assert_eq!(bus.stats().messages_delivered, 1);
    }

    #[test]
    fn test_stats_broadcasts_sent() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Broadcaster");
        bus.register_agent("agent-2", "Subscriber");
        bus.subscribe("agent-2", "topic");

        bus.broadcast("agent-1", "topic", json!({}));

        assert_eq!(bus.stats().broadcasts_sent, 1);
    }

    #[test]
    fn test_stats_events_published() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Source");

        bus.publish_event("agent-1", EventType::AgentStarted, json!({}));

        assert_eq!(bus.stats().events_published, 1);
    }

    #[test]
    fn test_stats_heartbeats_received() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("agent-1", "Agent");

        bus.heartbeat("agent-1", HeartbeatStatus::Healthy);
        bus.heartbeat("agent-1", HeartbeatStatus::Healthy);

        assert_eq!(bus.stats().heartbeats_received, 2);
    }

    // ------------------------------------------------------------------------
    // Message Type Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_message_request_creation() {
        let msg = Message::request("from".to_string(), "to".to_string(), json!({"key": "value"}));

        match msg {
            Message::Request { from, to, payload, .. } => {
                assert_eq!(from, "from");
                assert_eq!(to, "to");
                assert_eq!(payload["key"], "value");
            }
            _ => panic!("Expected Request message"),
        }
    }

    #[test]
    fn test_message_response_creation() {
        let msg = Message::response("req-123".to_string(), "to".to_string(), json!({}), true);

        match msg {
            Message::Response { id, success, .. } => {
                assert_eq!(id, "req-123");
                assert!(success);
            }
            _ => panic!("Expected Response message"),
        }
    }

    #[test]
    fn test_message_broadcast_creation() {
        let msg = Message::broadcast("from".to_string(), "topic".to_string(), json!({}));

        match msg {
            Message::Broadcast { from, topic, .. } => {
                assert_eq!(from, "from");
                assert_eq!(topic, "topic");
            }
            _ => panic!("Expected Broadcast message"),
        }
    }

    #[test]
    fn test_message_event_creation() {
        let msg = Message::event("source".to_string(), EventType::TaskCompleted, json!({}));

        match msg {
            Message::Event { source, event_type, .. } => {
                assert_eq!(source, "source");
                assert_eq!(event_type, EventType::TaskCompleted);
            }
            _ => panic!("Expected Event message"),
        }
    }

    #[test]
    fn test_message_heartbeat_creation() {
        let msg = Message::heartbeat("agent".to_string(), HeartbeatStatus::Healthy);

        match msg {
            Message::Heartbeat { agent, status, .. } => {
                assert_eq!(agent, "agent");
                assert_eq!(status, HeartbeatStatus::Healthy);
            }
            _ => panic!("Expected Heartbeat message"),
        }
    }

    #[test]
    fn test_message_id() {
        let request = Message::request("from".to_string(), "to".to_string(), json!({}));
        assert!(request.id().is_some());

        let broadcast = Message::broadcast("from".to_string(), "topic".to_string(), json!({}));
        assert!(broadcast.id().is_none());
    }

    #[test]
    fn test_message_source() {
        let request = Message::request("from".to_string(), "to".to_string(), json!({}));
        assert_eq!(request.source(), Some("from"));

        let response = Message::response("id".to_string(), "to".to_string(), json!({}), true);
        assert_eq!(response.source(), None);
    }

    #[test]
    fn test_message_is_heartbeat() {
        let heartbeat = Message::heartbeat("agent".to_string(), HeartbeatStatus::Healthy);
        assert!(heartbeat.is_heartbeat());

        let request = Message::request("from".to_string(), "to".to_string(), json!({}));
        assert!(!request.is_heartbeat());
    }

    #[test]
    fn test_message_is_event() {
        let event = Message::event("source".to_string(), EventType::AgentStarted, json!({}));
        assert!(event.is_event());

        let request = Message::request("from".to_string(), "to".to_string(), json!({}));
        assert!(!request.is_event());
    }

    #[test]
    fn test_message_timestamp() {
        let before = Utc::now();
        let msg = Message::request("from".to_string(), "to".to_string(), json!({}));
        let after = Utc::now();

        let ts = msg.timestamp();
        assert!(ts >= before && ts <= after);
    }

    // ------------------------------------------------------------------------
    // Agent Info Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_agent_info_new() {
        let info = AgentInfo::new("agent-1".to_string(), "Test Agent");

        assert_eq!(info.id, "agent-1");
        assert_eq!(info.name, "Test Agent");
        assert_eq!(info.status, AgentStatus::Active);
        assert!(info.last_heartbeat.is_none());
        assert!(info.subscriptions.is_empty());
    }

    #[test]
    fn test_agent_info_update_heartbeat() {
        let mut info = AgentInfo::new("agent-1".to_string(), "Test Agent");

        info.update_heartbeat(HeartbeatStatus::Healthy);
        assert!(info.last_heartbeat.is_some());
        assert_eq!(info.status, AgentStatus::Active);

        info.update_heartbeat(HeartbeatStatus::Unhealthy);
        assert_eq!(info.status, AgentStatus::Offline);
    }

    #[test]
    fn test_agent_info_is_healthy() {
        let mut info = AgentInfo::new("agent-1".to_string(), "Test Agent");

        // No heartbeat = not healthy
        assert!(!info.is_healthy(1000));

        // With heartbeat = healthy
        info.update_heartbeat(HeartbeatStatus::Healthy);
        assert!(info.is_healthy(1000));
    }

    // ------------------------------------------------------------------------
    // Event Type Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_event_type_display() {
        assert_eq!(format!("{}", EventType::AgentStarted), "AgentStarted");
        assert_eq!(format!("{}", EventType::TaskFailed), "TaskFailed");
        assert_eq!(
            format!("{}", EventType::Custom("MyEvent".to_string())),
            "Custom(MyEvent)"
        );
    }

    #[test]
    fn test_event_type_equality() {
        assert_eq!(EventType::AgentStarted, EventType::AgentStarted);
        assert_ne!(EventType::AgentStarted, EventType::AgentStopped);
        assert_eq!(
            EventType::Custom("A".to_string()),
            EventType::Custom("A".to_string())
        );
        assert_ne!(
            EventType::Custom("A".to_string()),
            EventType::Custom("B".to_string())
        );
    }

    // ------------------------------------------------------------------------
    // Heartbeat Status Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_heartbeat_status_default() {
        let status: HeartbeatStatus = Default::default();
        assert_eq!(status, HeartbeatStatus::Healthy);
    }

    // ------------------------------------------------------------------------
    // Agent Status Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_agent_status_default() {
        let status: AgentStatus = Default::default();
        assert_eq!(status, AgentStatus::Unknown);
    }

    // ------------------------------------------------------------------------
    // Bus Participant Trait Tests
    // ------------------------------------------------------------------------

    struct TestParticipant {
        id: AgentId,
        received_count: usize,
    }

    impl BusParticipant for TestParticipant {
        fn agent_id(&self) -> AgentId {
            self.id.clone()
        }

        fn on_message(&mut self, _msg: &Message) -> Option<Message> {
            self.received_count += 1;
            None
        }

        fn health_status(&self) -> HeartbeatStatus {
            HeartbeatStatus::Healthy
        }

        fn subscribed_topics(&self) -> Vec<Topic> {
            vec!["test-topic".to_string()]
        }
    }

    #[test]
    fn test_bus_participant_trait() {
        let mut participant = TestParticipant {
            id: "test-agent".to_string(),
            received_count: 0,
        };

        assert_eq!(participant.agent_id(), "test-agent");
        assert_eq!(participant.health_status(), HeartbeatStatus::Healthy);
        assert_eq!(participant.subscribed_topics(), vec!["test-topic"]);

        let msg = Message::request("from".to_string(), "to".to_string(), json!({}));
        let response = participant.on_message(&msg);
        assert!(response.is_none());
        assert_eq!(participant.received_count, 1);
    }

    // ------------------------------------------------------------------------
    // Integration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_full_request_response_flow() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("client", "Client Agent");
        bus.register_agent("server", "Server Agent");

        // Client sends request
        let req_id = bus.send("client", "server", json!({"action": "get_data"})).unwrap();

        // Server receives request
        let server_msgs = bus.poll_messages("server");
        assert_eq!(server_msgs.len(), 1);

        // Server sends response
        bus.respond(&req_id, "client", json!({"data": "result"}), true);

        // Client receives response
        let client_msgs = bus.poll_messages("client");
        assert_eq!(client_msgs.len(), 1);

        match &client_msgs[0] {
            Message::Response { id, success, payload, .. } => {
                assert_eq!(id, &req_id);
                assert!(success);
                assert_eq!(payload["data"], "result");
            }
            _ => panic!("Expected Response message"),
        }
    }

    #[test]
    fn test_pubsub_flow() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("publisher", "Publisher");
        bus.register_agent("sub-1", "Subscriber 1");
        bus.register_agent("sub-2", "Subscriber 2");
        bus.register_agent("non-sub", "Non-Subscriber");

        bus.subscribe("sub-1", "updates");
        bus.subscribe("sub-2", "updates");

        bus.broadcast("publisher", "updates", json!({"version": "1.0"}));

        assert_eq!(bus.pending_message_count("sub-1"), 1);
        assert_eq!(bus.pending_message_count("sub-2"), 1);
        assert_eq!(bus.pending_message_count("non-sub"), 0);
    }

    #[test]
    fn test_event_propagation_flow() {
        let mut bus = AgentBus::new(BusConfig::minimal());
        bus.register_agent("monitor", "System Monitor");
        bus.register_agent("logger", "Event Logger");
        bus.register_agent("alerter", "Alert Handler");

        // Publish system event
        bus.publish_event(
            "monitor",
            EventType::ResourceWarning,
            json!({"cpu": 95, "memory": 80}),
        );

        // All agents should receive the event
        let monitor_msgs = bus.poll_messages("monitor");
        let logger_msgs = bus.poll_messages("logger");
        let alerter_msgs = bus.poll_messages("alerter");

        assert!(!monitor_msgs.is_empty());
        assert!(!logger_msgs.is_empty());
        assert!(!alerter_msgs.is_empty());
    }

    #[test]
    fn test_health_monitoring_flow() {
        let mut bus = AgentBus::new(BusConfig {
            timeout_ms: 100,
            ..BusConfig::minimal()
        });

        bus.register_agent("healthy", "Healthy Agent");
        bus.register_agent("sick", "Sick Agent");

        // Healthy agent sends heartbeat
        bus.heartbeat("healthy", HeartbeatStatus::Healthy);

        // Check health
        let health = bus.check_health();

        let (_, is_healthy) = health.get("healthy").unwrap();
        assert!(is_healthy);

        let (_, is_sick_healthy) = health.get("sick").unwrap();
        assert!(!is_sick_healthy);
    }
}
