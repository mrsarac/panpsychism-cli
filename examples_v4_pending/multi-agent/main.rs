//! Example: Multi-Agent Orchestration
//!
//! This example demonstrates how to orchestrate multiple agents
//! using the AgentBus for complex workflows. Each agent specializes
//! in a specific task, and they communicate through the bus.
//!
//! Architecture:
//! ```text
//! +------------------+
//! |    AgentBus      |  <-- Central communication hub
//! +------------------+
//!          |
//!    +-----+-----+-----+-----+
//!    |     |     |     |     |
//!    v     v     v     v     v
//! +------+------+------+------+------+
//! |Enhanc|Format|Summar|Expand|Debug |
//! +------+------+------+------+------+
//! ```

use panpsychism::{
    bus::{AgentBus, AgentId, BusConfig, Message, Topic},
    config::Config,
    enhancer::{EnhancerAgentBuilder, EnhancerConfig},
    expander::{ExpanderAgentBuilder, ExpanderConfig},
    formatter::{FormatterAgentBuilder, OutputFormat},
    summarizer::{SummarizerAgent, SummarizerConfig},
    Result,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    panpsychism::setup_logging();

    println!("=== Panpsychism Multi-Agent Example ===\n");

    // Load configuration
    let config = Config::load()?;

    // === Part 1: Create the Agent Bus ===
    println!("--- Part 1: Agent Bus Setup ---\n");

    let bus_config = BusConfig::default();
    let bus = Arc::new(AgentBus::new(bus_config));

    // Register agents with the bus
    let enhancer_id = AgentId::new("enhancer");
    let formatter_id = AgentId::new("formatter");
    let summarizer_id = AgentId::new("summarizer");
    let expander_id = AgentId::new("expander");

    bus.register(enhancer_id.clone(), "EnhancerAgent").await?;
    bus.register(formatter_id.clone(), "FormatterAgent").await?;
    bus.register(summarizer_id.clone(), "SummarizerAgent").await?;
    bus.register(expander_id.clone(), "ExpanderAgent").await?;

    println!("Registered 4 agents with the bus");

    // === Part 2: Create Agents ===
    println!("\n--- Part 2: Agent Creation ---\n");

    // Enhancer Agent - improves prompt quality
    let enhancer_config = EnhancerConfig::default();
    let enhancer = EnhancerAgentBuilder::new()
        .with_config(enhancer_config)
        .build()?;
    println!("Created EnhancerAgent");

    // Formatter Agent - formats output
    let formatter = FormatterAgentBuilder::new()
        .with_default_format(OutputFormat::Markdown)
        .build()?;
    println!("Created FormatterAgent (Markdown output)");

    // Summarizer Agent - generates summaries
    let summarizer_config = SummarizerConfig::default();
    let summarizer = SummarizerAgent::new(summarizer_config);
    println!("Created SummarizerAgent");

    // Expander Agent - expands content with details
    let expander_config = ExpanderConfig::default();
    let expander = ExpanderAgentBuilder::new()
        .with_config(expander_config)
        .build()?;
    println!("Created ExpanderAgent");

    // === Part 3: Message Passing ===
    println!("\n--- Part 3: Message Passing ---\n");

    let workflow_topic = Topic::new("workflow");

    // Publish a message to the workflow topic
    let input = "Implement secure authentication with OAuth2 and JWT tokens";

    bus.publish(
        workflow_topic.clone(),
        Message::text("input", input),
    )
    .await?;
    println!("Published input to workflow topic");

    // Simulate agent processing by publishing results
    bus.publish(
        workflow_topic.clone(),
        Message::text("enhanced", "Enhanced: Implement a secure authentication system using OAuth2 for third-party identity providers and JWT tokens for stateless session management..."),
    )
    .await?;
    println!("EnhancerAgent published result");

    bus.publish(
        workflow_topic.clone(),
        Message::text("formatted", "## Authentication Implementation\n\n### OAuth2 Setup\n...\n\n### JWT Configuration\n..."),
    )
    .await?;
    println!("FormatterAgent published result");

    bus.publish(
        workflow_topic.clone(),
        Message::text("summarized", "Summary: Secure auth system using OAuth2 + JWT for stateless sessions."),
    )
    .await?;
    println!("SummarizerAgent published result");

    // === Part 4: Bus Statistics ===
    println!("\n--- Part 4: Bus Statistics ---\n");

    let stats = bus.stats().await;
    println!("Total messages: {}", stats.total_messages);
    println!("Active agents: {}", stats.active_agents);
    println!("Topics: {:?}", stats.topics);

    // === Part 5: Retrieve Messages ===
    println!("\n--- Part 5: Workflow Messages ---\n");

    let messages = bus.get_topic_messages(&workflow_topic).await;
    for msg in messages {
        println!("[{}] {}: {}...",
            msg.topic,
            msg.key,
            &msg.value[..50.min(msg.value.len())]
        );
    }

    Ok(())
}

/// Example: Parallel agent execution pattern
#[allow(dead_code)]
async fn parallel_processing_example() -> Result<()> {
    use futures::future::join_all;

    let enhancer_config = EnhancerConfig::default();

    // Create multiple enhancers for parallel processing
    let enhancers: Vec<_> = (0..4)
        .map(|_| {
            EnhancerAgentBuilder::new()
                .with_config(enhancer_config.clone())
                .build()
        })
        .collect::<Result<Vec<_>>>()?;

    let inputs = vec![
        "Explain ownership",
        "Describe lifetimes",
        "Define traits",
        "Explain async",
    ];

    // Process all inputs in parallel
    let futures: Vec<_> = enhancers
        .iter()
        .zip(inputs.iter())
        .map(|(enhancer, input)| enhancer.enhance(*input))
        .collect();

    let results = join_all(futures).await;

    println!("Processed {} inputs in parallel", results.len());
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(enhanced) => println!("  {}: {} chars", i, enhanced.content.len()),
            Err(e) => println!("  {}: Error - {}", i, e),
        }
    }

    Ok(())
}

/// Example: Chain of agents pattern
#[allow(dead_code)]
async fn chain_pattern_example() -> Result<()> {
    let enhancer = EnhancerAgentBuilder::new()
        .with_config(EnhancerConfig::default())
        .build()?;

    let formatter = FormatterAgentBuilder::new()
        .with_default_format(OutputFormat::Markdown)
        .build()?;

    let summarizer = SummarizerAgent::new(SummarizerConfig::default());

    // Chain: input -> enhance -> format -> summarize
    let input = "Explain Rust error handling";

    println!("Input: {}", input);

    let enhanced = enhancer.enhance(input).await?;
    println!("Enhanced: {} chars", enhanced.content.len());

    let formatted = formatter.format(&enhanced.content).await?;
    println!("Formatted: {} chars", formatted.content.len());

    let summary = summarizer.summarize(&formatted.content).await?;
    println!("Summary: {}", summary.content);

    Ok(())
}

/// Example: Agent health monitoring
#[allow(dead_code)]
async fn health_monitoring_example(bus: &AgentBus) -> Result<()> {
    use panpsychism::monitor::{MonitorAgentBuilder, MonitorConfig};

    let monitor_config = MonitorConfig::default();
    let monitor = MonitorAgentBuilder::new()
        .with_config(monitor_config)
        .build()?;

    // Get health report for all agents
    let report = monitor.health_report().await?;

    println!("=== Agent Health Report ===");
    println!("Overall status: {:?}", report.overall_status);

    for agent_health in &report.agents {
        println!(
            "  {}: {:?} (latency: {:?})",
            agent_health.id, agent_health.status, agent_health.avg_latency
        );
    }

    Ok(())
}
