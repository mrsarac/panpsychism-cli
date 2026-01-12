//! Example: Basic Query
//!
//! This example demonstrates the simplest way to use Panpsychism
//! for analyzing user intent and selecting optimal prompts.
//!
//! The Sorcerer's Wand metaphor:
//! - You (Sorcerer) speak an incantation (query)
//! - The Wand (Orchestrator) analyzes your intent
//! - Creation emerges (selected prompts and strategy)

use panpsychism::{
    config::Config,
    orchestrator::{Orchestrator, Strategy},
    privacy::PrivacyTier,
    search::{SearchEngine, SearchResult},
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    panpsychism::setup_logging();

    println!("=== Panpsychism Basic Query Example ===\n");

    // Load configuration from file or use defaults
    let config = Config::load()?;

    // Create the orchestrator (The Wand)
    let orchestrator = Orchestrator::new();

    // The user's incantation
    let query = "How do I implement authentication in a Rust web application?";

    println!("Query: {}\n", query);
    println!("Analyzing intent...\n");

    // Analyze the user's intent
    let intent = orchestrator.analyze_intent(query).await?;

    // Display the analysis
    println!("=== Intent Analysis ===");
    println!("Category: {:?}", intent.category);
    println!("Strategy: {:?}", intent.strategy);
    println!("Complexity: {}", intent.complexity);
    println!("Keywords: {:?}", intent.keywords);

    // Create search engine and find relevant prompts
    let search_engine = SearchEngine::new(&config.prompts_dir);

    // Search for relevant prompts based on the query
    let results = search_engine.search(query, 5)?;

    println!("\n=== Search Results ===");
    for (i, result) in results.iter().enumerate() {
        println!(
            "{}. {} (score: {:.2})",
            i + 1,
            result.prompt_id,
            result.score
        );
    }

    // Select prompts using the orchestrator
    let selected = orchestrator.select_prompts(&results, &intent)?;

    println!("\n=== Selected Prompts ===");
    println!("Strategy: {:?}", selected.strategy);
    println!("Prompts:");
    for prompt in &selected.prompts {
        println!("  - {} (role: {:?})", prompt.prompt_id, prompt.role);
    }

    Ok(())
}

// Alternative: With custom orchestrator settings
#[allow(dead_code)]
async fn custom_orchestrator_example() -> Result<()> {
    // Create orchestrator with custom limits
    let orchestrator = Orchestrator::with_limits(3, 10) // min 3, max 10 prompts
        .with_relevance_threshold(0.5); // Higher threshold

    let query = "Compare REST and GraphQL for a microservices architecture";

    let intent = orchestrator.analyze_intent(query).await?;
    println!("Strategy for comparison query: {:?}", intent.strategy);
    // Should be Strategy::Parallel for comparison queries

    Ok(())
}

// Alternative: Demonstrate different strategies
#[allow(dead_code)]
async fn strategy_examples() -> Result<()> {
    let orchestrator = Orchestrator::new();

    // Different query patterns trigger different strategies
    let queries = vec![
        ("What is Rust ownership?", "Simple query -> Focused"),
        ("Compare async vs threads in Rust", "Comparison -> Parallel"),
        ("First set up the project, then add tests, finally deploy", "Sequential -> Chain"),
        ("Explain authentication, authorization, and session management", "Broad topic -> Ensemble"),
    ];

    println!("=== Strategy Selection Examples ===\n");

    for (query, description) in queries {
        let intent = orchestrator.analyze_intent(query).await?;
        println!("{}", description);
        println!("  Query: {}", query);
        println!("  Strategy: {:?}", intent.strategy);
        println!("  Complexity: {}", intent.complexity);
        println!();
    }

    Ok(())
}

// Alternative: Privacy-aware processing
#[allow(dead_code)]
fn privacy_config_example() {
    // Default is LOCAL (maximum privacy)
    let default_privacy = PrivacyTier::default();
    assert_eq!(default_privacy, PrivacyTier::Local);

    // Different tiers available
    let tiers = [
        PrivacyTier::Local,    // All data stays on device
        PrivacyTier::Hybrid,   // User controls sharing
        PrivacyTier::Federated, // Full collaboration
    ];

    println!("=== Privacy Tiers ===");
    for tier in tiers {
        println!("{:?}", tier);
    }
}
