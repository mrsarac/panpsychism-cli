//! Integration tests for Agent Communication Bus (ACB)
//!
//! These tests verify the inter-agent communication patterns:
//! - Channel creation and management
//! - Point-to-point messaging
//! - Pub/Sub broadcast messaging
//! - Event propagation
//! - Heartbeat monitoring
//! - Error handling and edge cases
//! - Multi-agent orchestration scenarios

use panpsychism::bus::{
    AgentBus, AgentId, AgentStatus, BusConfig, BusParticipant, EventType, HeartbeatStatus,
    Message, Topic,
};
use serde_json::json;

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a bus with minimal configuration for fast tests.
fn create_test_bus() -> AgentBus {
    AgentBus::new(BusConfig::minimal())
}

/// Create a bus with custom queue size for overflow tests.
fn create_bus_with_queue_size(size: usize) -> AgentBus {
    AgentBus::new(BusConfig {
        max_queue_size: size,
        ..BusConfig::minimal()
    })
}

/// Create a bus with custom log size for ring buffer tests.
fn create_bus_with_log_size(size: usize) -> AgentBus {
    AgentBus::new(BusConfig {
        max_log_size: size,
        enable_logging: true,
        ..BusConfig::minimal()
    })
}

/// Create a bus with short timeout for health check tests.
fn create_bus_with_timeout(timeout_ms: u64) -> AgentBus {
    AgentBus::new(BusConfig {
        timeout_ms,
        ..BusConfig::minimal()
    })
}

/// Register multiple agents at once.
fn register_agents(bus: &mut AgentBus, ids: &[&str]) {
    for id in ids {
        bus.register_agent(*id, format!("Agent {}", id));
    }
}

/// A test participant implementation for trait tests.
struct MockParticipant {
    id: AgentId,
    received_messages: Vec<Message>,
    health: HeartbeatStatus,
    topics: Vec<Topic>,
}

impl MockParticipant {
    fn new(id: &str) -> Self {
        MockParticipant {
            id: id.to_string(),
            received_messages: Vec::new(),
            health: HeartbeatStatus::Healthy,
            topics: Vec::new(),
        }
    }

    fn with_topics(mut self, topics: Vec<&str>) -> Self {
        self.topics = topics.into_iter().map(String::from).collect();
        self
    }

    fn with_health(mut self, status: HeartbeatStatus) -> Self {
        self.health = status;
        self
    }

    #[allow(dead_code)]
    fn message_count(&self) -> usize {
        self.received_messages.len()
    }
}

impl BusParticipant for MockParticipant {
    fn agent_id(&self) -> AgentId {
        self.id.clone()
    }

    fn on_message(&mut self, msg: &Message) -> Option<Message> {
        self.received_messages.push(msg.clone());

        // Optionally respond to requests
        if let Message::Request { id, from, .. } = msg {
            Some(Message::response(
                id.clone(),
                from.clone(),
                json!({"status": "received"}),
                true,
            ))
        } else {
            None
        }
    }

    fn health_status(&self) -> HeartbeatStatus {
        self.health
    }

    fn subscribed_topics(&self) -> Vec<Topic> {
        self.topics.clone()
    }
}

// ============================================================================
// Channel Creation Tests
// ============================================================================

mod channel_creation_tests {
    use super::*;

    #[test]
    fn test_create_channel() {
        let mut bus = create_test_bus();

        // Register agents (which creates their message queues/channels)
        assert!(
            bus.register_agent("analyzer", "Analyzer Agent"),
            "Should create channel for first agent"
        );
        assert!(
            bus.register_agent("synthesizer", "Synthesizer Agent"),
            "Should create channel for second agent"
        );

        assert_eq!(bus.agent_count(), 2, "Should have 2 registered agents");
        assert!(bus.is_registered("analyzer"), "Analyzer should be registered");
        assert!(bus.is_registered("synthesizer"), "Synthesizer should be registered");
    }

    #[test]
    fn test_channel_capacity() {
        let mut bus = create_bus_with_queue_size(5);
        register_agents(&mut bus, &["sender", "receiver"]);

        // Send messages up to capacity
        for i in 0..5 {
            let msg_id = bus.send("sender", "receiver", json!({"index": i}));
            assert!(msg_id.is_some(), "Should send message {} within capacity", i);
        }

        assert_eq!(
            bus.pending_message_count("receiver"),
            5,
            "Should have 5 pending messages"
        );

        // Send beyond capacity - should still succeed but drop oldest
        let msg_id = bus.send("sender", "receiver", json!({"index": 5}));
        assert!(msg_id.is_some(), "Should accept message beyond capacity");
        assert_eq!(
            bus.pending_message_count("receiver"),
            5,
            "Queue should maintain max capacity"
        );
        assert_eq!(bus.stats().messages_dropped, 1, "Should have dropped 1 message");
    }

    #[test]
    fn test_duplicate_channel_handling() {
        let mut bus = create_test_bus();

        assert!(
            bus.register_agent("agent-1", "First Agent"),
            "First registration should succeed"
        );
        assert!(
            !bus.register_agent("agent-1", "Duplicate Agent"),
            "Duplicate registration should fail"
        );
        assert_eq!(bus.agent_count(), 1, "Should still have only 1 agent");

        // Verify original agent info is preserved
        let agent = bus.get_agent("agent-1").expect("Agent should exist");
        assert_eq!(agent.name, "First Agent", "Original name should be preserved");
    }

    #[test]
    fn test_channel_cleanup_on_unregister() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["sender", "receiver"]);

        // Send some messages
        bus.send("sender", "receiver", json!({"data": "test"}));
        assert_eq!(bus.pending_message_count("receiver"), 1);

        // Unregister receiver
        assert!(bus.unregister_agent("receiver"), "Unregister should succeed");
        assert_eq!(
            bus.pending_message_count("receiver"),
            0,
            "Queue should be cleaned up"
        );
        assert!(!bus.is_registered("receiver"), "Agent should be unregistered");
    }
}

// ============================================================================
// Message Passing Tests
// ============================================================================

mod message_passing_tests {
    use super::*;

    #[test]
    fn test_send_request_message() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["client", "server"]);

        let payload = json!({
            "action": "get_data",
            "params": {"key": "value"}
        });

        let msg_id = bus.send("client", "server", payload.clone());
        assert!(msg_id.is_some(), "Send should return message ID");

        let messages = bus.poll_messages("server");
        assert_eq!(messages.len(), 1, "Server should have 1 message");

        match &messages[0] {
            Message::Request { from, to, payload: msg_payload, .. } => {
                assert_eq!(from, "client", "From should be client");
                assert_eq!(to, "server", "To should be server");
                assert_eq!(msg_payload["action"], "get_data", "Payload should match");
            }
            _ => panic!("Expected Request message"),
        }
    }

    #[test]
    fn test_broadcast_message() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["publisher", "sub-1", "sub-2", "sub-3"]);

        // Subscribe agents to topic
        bus.subscribe("sub-1", "news");
        bus.subscribe("sub-2", "news");
        bus.subscribe("sub-3", "news");

        let delivered = bus.broadcast("publisher", "news", json!({"headline": "Breaking!"}));
        assert_eq!(delivered, 3, "Should deliver to all 3 subscribers");

        for sub in &["sub-1", "sub-2", "sub-3"] {
            let messages = bus.poll_messages(sub);
            assert_eq!(messages.len(), 1, "{} should have 1 message", sub);

            match &messages[0] {
                Message::Broadcast { from, topic, .. } => {
                    assert_eq!(from, "publisher", "From should be publisher");
                    assert_eq!(topic, "news", "Topic should be news");
                }
                _ => panic!("Expected Broadcast message"),
            }
        }
    }

    #[test]
    fn test_event_emission() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["source", "listener-1", "listener-2"]);

        let success = bus.publish_event(
            "source",
            EventType::TaskCompleted,
            json!({"task_id": "task-123", "duration_ms": 500}),
        );
        assert!(success, "Event publication should succeed");

        // All agents (including source) should receive the event
        for agent in &["source", "listener-1", "listener-2"] {
            let messages = bus.poll_messages(agent);
            assert!(!messages.is_empty(), "{} should receive the event", agent);

            let event = messages.iter().find(|m| m.is_event());
            assert!(event.is_some(), "{} should have event message", agent);

            if let Some(Message::Event { event_type, data, .. }) = event {
                assert_eq!(*event_type, EventType::TaskCompleted);
                assert_eq!(data["task_id"], "task-123");
            }
        }
    }

    #[test]
    fn test_heartbeat_message() {
        let mut bus = AgentBus::new(BusConfig {
            enable_logging: true,
            ..BusConfig::minimal()
        });
        register_agents(&mut bus, &["worker"]);

        assert!(bus.heartbeat("worker", HeartbeatStatus::Healthy), "Heartbeat should succeed");

        let agent = bus.get_agent("worker").expect("Agent should exist");
        assert!(agent.last_heartbeat.is_some(), "Should record heartbeat time");
        assert_eq!(agent.status, AgentStatus::Active, "Status should be Active");

        // Check stats
        assert_eq!(bus.stats().heartbeats_received, 1, "Should count heartbeat");
    }

    #[test]
    fn test_response_message() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["requester", "responder"]);

        // Send request
        let req_id = bus
            .send("requester", "responder", json!({"query": "data"}))
            .expect("Should get request ID");

        // Clear responder's queue (simulating processing)
        bus.poll_messages("responder");

        // Send response
        let success = bus.respond(&req_id, "requester", json!({"result": "success"}), true);
        assert!(success, "Response should succeed");

        // Requester should receive response
        let messages = bus.poll_messages("requester");
        assert_eq!(messages.len(), 1, "Requester should have response");

        match &messages[0] {
            Message::Response { id, success, payload, .. } => {
                assert_eq!(id, &req_id, "Response ID should match request");
                assert!(success, "Success flag should be true");
                assert_eq!(payload["result"], "success");
            }
            _ => panic!("Expected Response message"),
        }
    }

    #[test]
    fn test_message_ordering() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["sender", "receiver"]);

        // Send multiple messages
        for i in 0..5 {
            bus.send("sender", "receiver", json!({"order": i}));
        }

        let messages = bus.poll_messages("receiver");
        assert_eq!(messages.len(), 5, "Should receive all messages");

        // Verify FIFO ordering
        for (i, msg) in messages.iter().enumerate() {
            if let Message::Request { payload, .. } = msg {
                assert_eq!(
                    payload["order"].as_i64().unwrap() as usize,
                    i,
                    "Messages should be in order"
                );
            }
        }
    }
}

// ============================================================================
// Subscription Tests
// ============================================================================

mod subscription_tests {
    use super::*;

    #[test]
    fn test_topic_subscription() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["agent-1", "agent-2"]);

        assert!(bus.subscribe("agent-1", "topic-a"), "Subscription should succeed");
        assert!(bus.subscribe("agent-1", "topic-b"), "Second subscription should succeed");

        let subs = bus.get_agent_subscriptions("agent-1");
        assert_eq!(subs.len(), 2, "Agent should have 2 subscriptions");
        assert!(subs.contains(&"topic-a"), "Should be subscribed to topic-a");
        assert!(subs.contains(&"topic-b"), "Should be subscribed to topic-b");
    }

    #[test]
    fn test_multiple_subscribers() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["sub-1", "sub-2", "sub-3", "non-sub"]);

        // Subscribe some agents
        bus.subscribe("sub-1", "shared-topic");
        bus.subscribe("sub-2", "shared-topic");
        bus.subscribe("sub-3", "shared-topic");

        let subscribers = bus.get_subscribers("shared-topic");
        assert_eq!(subscribers.len(), 3, "Should have 3 subscribers");
        assert!(!subscribers.contains(&"non-sub"), "Non-subscriber should not be included");

        // Verify all subscribers receive broadcasts
        bus.register_agent("broadcaster", "Broadcaster");
        bus.broadcast("broadcaster", "shared-topic", json!({"msg": "hello all"}));

        for sub in &["sub-1", "sub-2", "sub-3"] {
            assert_eq!(
                bus.pending_message_count(sub),
                1,
                "{} should have broadcast message",
                sub
            );
        }
        assert_eq!(
            bus.pending_message_count("non-sub"),
            0,
            "Non-subscriber should not receive broadcast"
        );
    }

    #[test]
    fn test_unsubscribe() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["agent", "broadcaster"]);

        bus.subscribe("agent", "updates");
        assert_eq!(bus.get_subscribers("updates").len(), 1);

        // Broadcast before unsubscribe
        bus.broadcast("broadcaster", "updates", json!({"version": 1}));
        assert_eq!(bus.pending_message_count("agent"), 1);

        // Unsubscribe
        assert!(bus.unsubscribe("agent", "updates"), "Unsubscribe should succeed");
        assert!(bus.get_subscribers("updates").is_empty(), "No subscribers after unsubscribe");

        // Broadcast after unsubscribe
        bus.poll_messages("agent"); // Clear queue
        bus.broadcast("broadcaster", "updates", json!({"version": 2}));
        assert_eq!(
            bus.pending_message_count("agent"),
            0,
            "Should not receive after unsubscribe"
        );
    }

    #[test]
    fn test_subscription_persists_through_messages() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["subscriber", "publisher"]);

        bus.subscribe("subscriber", "events");

        // Send multiple broadcasts
        for i in 0..3 {
            bus.broadcast("publisher", "events", json!({"count": i}));
        }

        assert_eq!(
            bus.pending_message_count("subscriber"),
            3,
            "Should receive all broadcasts"
        );

        // Poll messages and verify subscription still active
        bus.poll_messages("subscriber");
        let subs = bus.get_agent_subscriptions("subscriber");
        assert!(subs.contains(&"events"), "Subscription should persist after poll");
    }

    #[test]
    fn test_cross_topic_subscriptions() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["agent", "pub-1", "pub-2"]);

        bus.subscribe("agent", "topic-a");
        bus.subscribe("agent", "topic-b");

        bus.broadcast("pub-1", "topic-a", json!({"from": "a"}));
        bus.broadcast("pub-2", "topic-b", json!({"from": "b"}));

        let messages = bus.poll_messages("agent");
        assert_eq!(messages.len(), 2, "Should receive from both topics");
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[test]
    fn test_send_to_nonexistent_agent() {
        let mut bus = create_test_bus();
        bus.register_agent("sender", "Sender");

        let result = bus.send("sender", "nonexistent", json!({}));
        assert!(
            result.is_none(),
            "Send to nonexistent agent should return None"
        );
    }

    #[test]
    fn test_send_from_nonexistent_agent() {
        let mut bus = create_test_bus();
        bus.register_agent("receiver", "Receiver");

        let result = bus.send("nonexistent", "receiver", json!({}));
        assert!(
            result.is_none(),
            "Send from nonexistent agent should return None"
        );
    }

    #[test]
    fn test_channel_overflow() {
        let mut bus = create_bus_with_queue_size(3);
        register_agents(&mut bus, &["sender", "receiver"]);

        // Fill the queue beyond capacity
        for i in 0..10 {
            bus.send("sender", "receiver", json!({"index": i}));
        }

        // Queue should contain only last 3 messages (indices 7, 8, 9)
        let messages = bus.poll_messages("receiver");
        assert_eq!(messages.len(), 3, "Queue should respect max size");

        // Verify oldest messages were dropped
        let indices: Vec<i64> = messages
            .iter()
            .filter_map(|m| {
                if let Message::Request { payload, .. } = m {
                    payload["index"].as_i64()
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(indices, vec![7, 8, 9], "Should keep newest messages");
        assert_eq!(bus.stats().messages_dropped, 7, "Should have dropped 7 messages");
    }

    #[test]
    fn test_broadcast_to_empty_topic() {
        let mut bus = create_test_bus();
        bus.register_agent("broadcaster", "Broadcaster");

        let delivered = bus.broadcast("broadcaster", "empty-topic", json!({}));
        assert_eq!(delivered, 0, "Should return 0 for empty topic");
    }

    #[test]
    fn test_subscribe_nonexistent_agent() {
        let mut bus = create_test_bus();

        let result = bus.subscribe("nonexistent", "topic");
        assert!(!result, "Subscribe should fail for nonexistent agent");
    }

    #[test]
    fn test_unsubscribe_nonexistent_agent() {
        let mut bus = create_test_bus();

        let result = bus.unsubscribe("nonexistent", "topic");
        assert!(!result, "Unsubscribe should fail for nonexistent agent");
    }

    #[test]
    fn test_heartbeat_nonexistent_agent() {
        let mut bus = create_test_bus();

        let result = bus.heartbeat("nonexistent", HeartbeatStatus::Healthy);
        assert!(!result, "Heartbeat should fail for nonexistent agent");
    }

    #[test]
    fn test_respond_to_nonexistent_requester() {
        let mut bus = create_test_bus();
        bus.register_agent("responder", "Responder");

        let result = bus.respond("some-request-id", "nonexistent", json!({}), true);
        assert!(!result, "Respond should fail for nonexistent requester");
    }

    #[test]
    fn test_poll_nonexistent_agent() {
        let mut bus = create_test_bus();

        let messages = bus.poll_messages("nonexistent");
        assert!(messages.is_empty(), "Poll should return empty for nonexistent agent");
    }

    #[test]
    fn test_event_from_nonexistent_source() {
        let mut bus = create_test_bus();

        let result = bus.publish_event("nonexistent", EventType::AgentStarted, json!({}));
        assert!(!result, "Event should fail from nonexistent source");
    }
}

// ============================================================================
// Integration Tests (Multi-Agent Scenarios)
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_multi_agent_communication() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["orchestrator", "analyzer", "synthesizer", "validator"]);

        // Orchestrator sends tasks to workers
        bus.send("orchestrator", "analyzer", json!({"task": "analyze", "data": "input"}));
        bus.send("orchestrator", "synthesizer", json!({"task": "synthesize", "context": {}}));
        bus.send("orchestrator", "validator", json!({"task": "validate", "rules": []}));

        // Verify each worker received their task
        assert_eq!(bus.pending_message_count("analyzer"), 1);
        assert_eq!(bus.pending_message_count("synthesizer"), 1);
        assert_eq!(bus.pending_message_count("validator"), 1);

        // Workers process and respond
        for worker in &["analyzer", "synthesizer", "validator"] {
            let messages = bus.poll_messages(worker);
            if let Message::Request { id, .. } = &messages[0] {
                bus.respond(id, "orchestrator", json!({"status": "done"}), true);
            }
        }

        // Orchestrator receives all responses
        let responses = bus.poll_messages("orchestrator");
        assert_eq!(responses.len(), 3, "Orchestrator should receive 3 responses");
    }

    #[test]
    fn test_message_routing() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["router", "handler-a", "handler-b", "handler-c"]);

        // Subscribe handlers to different topics
        bus.subscribe("handler-a", "type-a");
        bus.subscribe("handler-b", "type-b");
        bus.subscribe("handler-c", "type-c");

        // Router broadcasts to specific topics
        bus.broadcast("router", "type-a", json!({"route": "a"}));
        bus.broadcast("router", "type-b", json!({"route": "b"}));
        bus.broadcast("router", "type-c", json!({"route": "c"}));

        // Verify each handler only gets their routed message
        let a_msgs = bus.poll_messages("handler-a");
        let b_msgs = bus.poll_messages("handler-b");
        let c_msgs = bus.poll_messages("handler-c");

        assert_eq!(a_msgs.len(), 1);
        assert_eq!(b_msgs.len(), 1);
        assert_eq!(c_msgs.len(), 1);

        if let Message::Broadcast { payload, .. } = &a_msgs[0] {
            assert_eq!(payload["route"], "a");
        }
        if let Message::Broadcast { payload, .. } = &b_msgs[0] {
            assert_eq!(payload["route"], "b");
        }
        if let Message::Broadcast { payload, .. } = &c_msgs[0] {
            assert_eq!(payload["route"], "c");
        }
    }

    #[test]
    fn test_bus_shutdown_cleanup() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["agent-1", "agent-2", "agent-3"]);

        // Setup subscriptions and send messages
        bus.subscribe("agent-1", "shared");
        bus.subscribe("agent-2", "shared");
        bus.send("agent-1", "agent-2", json!({}));
        bus.broadcast("agent-1", "shared", json!({}));

        // Unregister all agents (simulating shutdown)
        for agent in &["agent-1", "agent-2", "agent-3"] {
            bus.unregister_agent(agent);
        }

        assert_eq!(bus.agent_count(), 0, "All agents should be removed");
        assert!(bus.get_topics().is_empty() || bus.get_subscribers("shared").is_empty(),
            "Subscriptions should be cleaned up");
    }

    #[test]
    fn test_pipeline_communication_pattern() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["input", "stage-1", "stage-2", "stage-3", "output"]);

        // Pipeline: input -> stage-1 -> stage-2 -> stage-3 -> output
        let stages = ["input", "stage-1", "stage-2", "stage-3", "output"];

        // Send through pipeline
        for i in 0..stages.len() - 1 {
            bus.send(stages[i], stages[i + 1], json!({
                "stage": i,
                "data": format!("processed by {}", stages[i])
            }));
        }

        // Verify each stage received its input
        assert_eq!(bus.pending_message_count("stage-1"), 1);
        assert_eq!(bus.pending_message_count("stage-2"), 1);
        assert_eq!(bus.pending_message_count("stage-3"), 1);
        assert_eq!(bus.pending_message_count("output"), 1);
    }

    #[test]
    fn test_fanout_fanin_pattern() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["master", "worker-1", "worker-2", "worker-3", "aggregator"]);

        // Fan-out: master sends to all workers
        for i in 1..=3 {
            bus.send("master", &format!("worker-{}", i), json!({"task_id": i}));
        }

        // Workers process and send to aggregator
        for i in 1..=3 {
            let worker = format!("worker-{}", i);
            let messages = bus.poll_messages(&worker);
            if let Message::Request { payload, .. } = &messages[0] {
                bus.send(
                    &worker,
                    "aggregator",
                    json!({"result": payload["task_id"].as_i64().unwrap() * 2}),
                );
            }
        }

        // Fan-in: aggregator receives all results
        let results = bus.poll_messages("aggregator");
        assert_eq!(results.len(), 3, "Aggregator should receive from all workers");
    }

    #[test]
    fn test_event_driven_coordination() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["coordinator", "worker-a", "worker-b", "monitor"]);

        // Coordinator starts workers via events
        bus.publish_event("coordinator", EventType::AgentStarted, json!({"workers": ["a", "b"]}));

        // All agents should receive the event
        assert!(bus.pending_message_count("worker-a") > 0);
        assert!(bus.pending_message_count("worker-b") > 0);
        assert!(bus.pending_message_count("monitor") > 0);

        // Workers report completion via events
        bus.poll_messages("worker-a");
        bus.poll_messages("worker-b");

        bus.publish_event("worker-a", EventType::TaskCompleted, json!({"worker": "a"}));
        bus.publish_event("worker-b", EventType::TaskCompleted, json!({"worker": "b"}));

        // Monitor receives completion events
        let monitor_msgs = bus.poll_messages("monitor");
        let completion_events: Vec<_> = monitor_msgs
            .iter()
            .filter(|m| {
                if let Message::Event { event_type, .. } = m {
                    *event_type == EventType::TaskCompleted
                } else {
                    false
                }
            })
            .collect();

        assert_eq!(completion_events.len(), 2, "Monitor should see both completions");
    }

    #[test]
    fn test_health_monitoring_scenario() {
        let mut bus = create_bus_with_timeout(1000);
        register_agents(&mut bus, &["healthy-1", "healthy-2", "unhealthy"]);

        // Healthy agents send heartbeats
        bus.heartbeat("healthy-1", HeartbeatStatus::Healthy);
        bus.heartbeat("healthy-2", HeartbeatStatus::Healthy);
        // unhealthy agent doesn't send heartbeat

        let health = bus.check_health();

        let (_, is_h1_healthy) = health.get("healthy-1").unwrap();
        let (_, is_h2_healthy) = health.get("healthy-2").unwrap();
        let (_, is_unhealthy_healthy) = health.get("unhealthy").unwrap();

        assert!(is_h1_healthy, "healthy-1 should be healthy");
        assert!(is_h2_healthy, "healthy-2 should be healthy");
        assert!(!is_unhealthy_healthy, "unhealthy should not be healthy");

        let unhealthy_agents = bus.get_unhealthy_agents();
        assert_eq!(unhealthy_agents.len(), 1);
        assert_eq!(unhealthy_agents[0].id, "unhealthy");
    }

    #[test]
    fn test_message_log_integration() {
        let mut bus = create_bus_with_log_size(10);
        register_agents(&mut bus, &["sender", "receiver"]);

        // Send multiple messages
        for i in 0..15 {
            bus.send("sender", "receiver", json!({"index": i}));
        }

        // Log should have ring buffer behavior
        let log = bus.get_message_log(20);
        assert!(
            log.len() <= 10,
            "Log should respect max size of 10, got {}",
            log.len()
        );

        // Clear log
        bus.clear_message_log();
        assert_eq!(bus.message_log_size(), 0, "Log should be empty after clear");
    }

    #[test]
    fn test_bus_participant_trait_integration() {
        let mut bus = create_test_bus();

        let participant = MockParticipant::new("mock-agent")
            .with_topics(vec!["topic-1", "topic-2"])
            .with_health(HeartbeatStatus::Healthy);

        // Register using trait methods
        let agent_id = participant.agent_id();
        bus.register_agent(&agent_id, "Mock Agent");

        for topic in participant.subscribed_topics() {
            bus.subscribe(&agent_id, &topic);
        }

        bus.heartbeat(&agent_id, participant.health_status());

        // Verify integration
        assert!(bus.is_registered(&agent_id));
        assert_eq!(bus.get_agent_subscriptions(&agent_id).len(), 2);

        let agent = bus.get_agent(&agent_id).unwrap();
        assert!(agent.last_heartbeat.is_some());
    }

    #[test]
    fn test_statistics_accumulation() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["a", "b"]);
        bus.subscribe("b", "topic");

        // Perform various operations
        bus.send("a", "b", json!({}));
        bus.send("a", "b", json!({}));
        bus.broadcast("a", "topic", json!({}));
        bus.publish_event("a", EventType::AgentStarted, json!({}));
        bus.heartbeat("a", HeartbeatStatus::Healthy);

        // Poll to count deliveries
        bus.poll_messages("b");
        bus.poll_messages("a");

        let stats = bus.stats();
        assert!(stats.messages_sent > 0, "Should count messages sent");
        assert!(stats.messages_delivered > 0, "Should count messages delivered");
        assert_eq!(stats.broadcasts_sent, 1, "Should count broadcasts");
        assert_eq!(stats.events_published, 1, "Should count events");
        assert_eq!(stats.heartbeats_received, 1, "Should count heartbeats");
    }

    #[test]
    fn test_concurrent_topic_operations() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["agent-1", "agent-2", "agent-3"]);

        // Multiple agents subscribe to same topic
        for agent in &["agent-1", "agent-2", "agent-3"] {
            bus.subscribe(agent, "shared-topic");
        }

        // One agent unsubscribes
        bus.unsubscribe("agent-2", "shared-topic");

        // Broadcast
        bus.register_agent("broadcaster", "Broadcaster");
        let delivered = bus.broadcast("broadcaster", "shared-topic", json!({}));

        assert_eq!(delivered, 2, "Should deliver to remaining 2 subscribers");
        assert_eq!(bus.pending_message_count("agent-1"), 1);
        assert_eq!(bus.pending_message_count("agent-2"), 0);
        assert_eq!(bus.pending_message_count("agent-3"), 1);
    }
}

// ============================================================================
// Event Type Tests
// ============================================================================

mod event_type_tests {
    use super::*;

    #[test]
    fn test_all_event_types() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["source", "listener"]);

        let event_types = vec![
            EventType::AgentStarted,
            EventType::AgentStopped,
            EventType::AgentError,
            EventType::TaskCompleted,
            EventType::TaskFailed,
            EventType::ResourceWarning,
            EventType::ConfigChanged,
            EventType::Custom("MyCustomEvent".to_string()),
        ];

        for event_type in event_types {
            bus.publish_event("source", event_type.clone(), json!({"event": format!("{}", event_type)}));
        }

        let messages = bus.poll_messages("listener");
        assert_eq!(messages.len(), 8, "Should receive all 8 event types");

        // Verify custom event
        let custom_event = messages.iter().find(|m| {
            if let Message::Event { event_type: EventType::Custom(name), .. } = m {
                name == "MyCustomEvent"
            } else {
                false
            }
        });
        assert!(custom_event.is_some(), "Should find custom event");
    }

    #[test]
    fn test_heartbeat_status_transitions() {
        let mut bus = create_test_bus();
        bus.register_agent("agent", "Test Agent");

        // Healthy -> Active
        bus.heartbeat("agent", HeartbeatStatus::Healthy);
        assert_eq!(bus.get_agent("agent").unwrap().status, AgentStatus::Active);

        // Degraded -> still Active
        bus.heartbeat("agent", HeartbeatStatus::Degraded);
        assert_eq!(bus.get_agent("agent").unwrap().status, AgentStatus::Active);

        // Unhealthy -> Offline
        bus.heartbeat("agent", HeartbeatStatus::Unhealthy);
        assert_eq!(bus.get_agent("agent").unwrap().status, AgentStatus::Offline);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_payload() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["sender", "receiver"]);

        let msg_id = bus.send("sender", "receiver", json!({}));
        assert!(msg_id.is_some(), "Empty payload should be valid");

        let messages = bus.poll_messages("receiver");
        if let Message::Request { payload, .. } = &messages[0] {
            assert_eq!(*payload, json!({}), "Empty payload should be preserved");
        }
    }

    #[test]
    fn test_large_payload() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["sender", "receiver"]);

        // Create a large payload
        let large_data: Vec<i32> = (0..1000).collect();
        let payload = json!({"data": large_data});

        let msg_id = bus.send("sender", "receiver", payload.clone());
        assert!(msg_id.is_some(), "Large payload should be accepted");

        let messages = bus.poll_messages("receiver");
        if let Message::Request { payload: recv_payload, .. } = &messages[0] {
            assert_eq!(
                recv_payload["data"].as_array().unwrap().len(),
                1000,
                "Large payload should be preserved"
            );
        }
    }

    #[test]
    fn test_special_characters_in_ids() {
        let mut bus = create_test_bus();

        let special_ids = vec![
            "agent-with-dashes",
            "agent_with_underscores",
            "agent.with.dots",
            "agent:with:colons",
            "agent/with/slashes",
        ];

        for id in &special_ids {
            assert!(bus.register_agent(*id, format!("Agent {}", id)), "Should accept ID: {}", id);
        }

        // Verify all registered
        for id in &special_ids {
            assert!(bus.is_registered(id), "Should be registered: {}", id);
        }
    }

    #[test]
    fn test_unicode_in_payload() {
        let mut bus = create_test_bus();
        register_agents(&mut bus, &["sender", "receiver"]);

        let unicode_payload = json!({
            "message": "Hello 世界 🌍 مرحبا",
            "emoji": "🎉🎊🎁",
            "arabic": "مرحبا بالعالم"
        });

        bus.send("sender", "receiver", unicode_payload.clone());

        let messages = bus.poll_messages("receiver");
        if let Message::Request { payload, .. } = &messages[0] {
            assert_eq!(payload["message"], "Hello 世界 🌍 مرحبا");
            assert_eq!(payload["emoji"], "🎉🎊🎁");
        }
    }

    #[test]
    fn test_rapid_register_unregister() {
        let mut bus = create_test_bus();

        for i in 0..100 {
            let id = format!("agent-{}", i);
            bus.register_agent(&id, &id);
            if i % 2 == 0 {
                bus.unregister_agent(&id);
            }
        }

        assert_eq!(bus.agent_count(), 50, "Should have 50 agents remaining");
    }

    #[test]
    fn test_self_send() {
        let mut bus = create_test_bus();
        bus.register_agent("self-talker", "Self Talker");

        let msg_id = bus.send("self-talker", "self-talker", json!({"self": true}));
        assert!(msg_id.is_some(), "Self-send should work");

        let messages = bus.poll_messages("self-talker");
        assert_eq!(messages.len(), 1, "Should receive self-sent message");
    }

    #[test]
    fn test_broadcast_to_self_excluded() {
        let mut bus = create_test_bus();
        bus.register_agent("broadcaster", "Broadcaster");
        bus.subscribe("broadcaster", "topic");

        let delivered = bus.broadcast("broadcaster", "topic", json!({}));
        assert_eq!(delivered, 0, "Broadcaster should not receive own broadcast");
        assert_eq!(bus.pending_message_count("broadcaster"), 0);
    }
}
