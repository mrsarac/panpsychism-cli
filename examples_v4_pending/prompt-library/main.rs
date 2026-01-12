//! Example: Prompt Library
//!
//! This example demonstrates how to use the prompt library system
//! including the PromptStore, PromptSelector, and Templater for
//! dynamic prompt management.
//!
//! The Grimoire Metaphor:
//! - PromptStore is your spellbook collection
//! - PromptSelector finds the right spell
//! - Templater fills in the magical variables

use panpsychism::{
    config::Config,
    prompt_selector::PromptSelector,
    prompt_store::{ContentCategory, MasterPrompt, PromptExample, PromptStore},
    templater::{
        PromptTemplate, TemplateRegistry, TemplateVariable, TemplaterAgentBuilder, VariableType,
    },
    Result,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    panpsychism::setup_logging();

    println!("=== Panpsychism Prompt Library Example ===\n");

    // Load configuration
    let config = Config::load()?;

    // === Part 1: Using PromptStore ===
    println!("--- Part 1: PromptStore ---\n");

    // Initialize the prompt store (loads from prompts/ directory)
    let store = PromptStore::new(&config.prompts_dir)?;

    // List available prompts
    println!("Available prompts: {}", store.len());
    for prompt in store.list().take(5) {
        println!("  - {} ({:?})", prompt.id, prompt.category);
    }

    // Get a specific prompt by ID
    if let Some(prompt) = store.get("code-review") {
        println!("\nPrompt 'code-review':");
        println!("  Title: {}", prompt.title);
        println!("  Category: {:?}", prompt.category);
        println!("  Variables: {:?}", prompt.variables);
    }

    // === Part 2: Using PromptSelector ===
    println!("\n--- Part 2: PromptSelector ---\n");

    // Create a selector from the store
    let selector = PromptSelector::new(&store);

    // Select prompts based on user intent
    let query = "I need help reviewing my authentication code";
    let selection = selector.select(query, 3)?;

    println!("Query: {}", query);
    println!("Selected {} prompts:", selection.prompts.len());
    for (i, prompt) in selection.prompts.iter().enumerate() {
        println!(
            "  {}. {} (confidence: {:.2})",
            i + 1,
            prompt.id,
            prompt.confidence
        );
    }

    // === Part 3: Using Templater ===
    println!("\n--- Part 3: Templater ---\n");

    // Create a templater agent
    let templater = TemplaterAgentBuilder::new().build()?;

    // Define a template
    let template = PromptTemplate::new(
        "code-analysis",
        r#"
Analyze the following {{language}} code for {{focus_area}}:

```{{language}}
{{code}}
```

Please provide:
1. Overview of the code structure
2. Potential issues or improvements
3. Best practices recommendations
"#,
    )
    .with_variable(TemplateVariable::new("language", VariableType::String).required())
    .with_variable(TemplateVariable::new("code", VariableType::Text).required())
    .with_variable(
        TemplateVariable::new("focus_area", VariableType::String)
            .with_default("general quality"),
    );

    // Create variables
    let mut variables = HashMap::new();
    variables.insert("language".to_string(), "rust".to_string());
    variables.insert(
        "code".to_string(),
        r#"
fn process(data: Vec<u8>) -> Result<String, Error> {
    let s = String::from_utf8(data)?;
    Ok(s.trim().to_uppercase())
}
"#
        .to_string(),
    );
    variables.insert("focus_area".to_string(), "error handling".to_string());

    // Instantiate the template
    let instantiated = templater.instantiate(&template, &variables)?;

    println!("Template: {}", template.id);
    println!("Variables: {:?}", variables.keys().collect::<Vec<_>>());
    println!("\nInstantiated prompt:\n{}", instantiated.content);

    // === Part 4: Template Registry ===
    println!("\n--- Part 4: Template Registry ---\n");

    // Create a registry and register templates
    let mut registry = TemplateRegistry::new();

    registry.register(template)?;
    registry.register(
        PromptTemplate::new(
            "explain-concept",
            "Explain {{concept}} in {{style}} terms for someone with {{expertise_level}} expertise.",
        )
        .with_variable(TemplateVariable::new("concept", VariableType::String).required())
        .with_variable(
            TemplateVariable::new("style", VariableType::String).with_default("simple"),
        )
        .with_variable(
            TemplateVariable::new("expertise_level", VariableType::String).with_default("beginner"),
        ),
    )?;

    println!("Registered templates: {}", registry.len());
    for id in registry.list() {
        println!("  - {}", id);
    }

    // Retrieve and use a template
    if let Some(template) = registry.get("explain-concept") {
        let mut vars = HashMap::new();
        vars.insert("concept".to_string(), "async/await".to_string());

        let result = templater.instantiate(template, &vars)?;
        println!("\nInstantiated 'explain-concept':\n{}", result.content);
    }

    Ok(())
}

/// Example: Creating a custom MasterPrompt
#[allow(dead_code)]
fn create_custom_prompt() -> MasterPrompt {
    MasterPrompt::builder("custom-review")
        .title("Custom Code Review")
        .category(ContentCategory::Technical)
        .content(
            r#"
Review the provided code focusing on:
- Security vulnerabilities
- Performance optimizations
- Code readability
- Best practices

{{code}}
"#,
        )
        .variable("code")
        .example(PromptExample::new(
            "Review this function",
            "fn add(a: i32, b: i32) -> i32 { a + b }",
        ))
        .build()
}

/// Example: Prompt selection with category filter
#[allow(dead_code)]
fn select_by_category(
    store: &PromptStore,
    query: &str,
    category: ContentCategory,
) -> Result<Vec<String>> {
    let selector = PromptSelector::new(store);

    let selection = selector.select_with_filter(query, 5, |p| p.category == category)?;

    Ok(selection.prompts.iter().map(|p| p.id.clone()).collect())
}

/// Example: Dynamic template from user input
#[allow(dead_code)]
fn dynamic_template_example(user_template: &str) -> Result<String> {
    let templater = TemplaterAgentBuilder::new().build()?;

    let template = PromptTemplate::new("user-template", user_template);

    // Parse and validate the template
    let variable_names = template.extract_variables();
    println!("Found variables: {:?}", variable_names);

    // Would need user input for variables in real usage
    let vars: HashMap<String, String> = HashMap::new();
    let result = templater.instantiate(&template, &vars)?;

    Ok(result.content)
}

/// Example: Batch template instantiation
#[allow(dead_code)]
fn batch_instantiation_example() -> Result<Vec<String>> {
    let templater = TemplaterAgentBuilder::new().build()?;

    let template =
        PromptTemplate::new("greeting", "Hello, {{name}}! Welcome to {{place}}.");

    let data = vec![
        [("name", "Alice"), ("place", "Rust")],
        [("name", "Bob"), ("place", "Panpsychism")],
        [("name", "Charlie"), ("place", "AGI")],
    ];

    let results: Vec<String> = data
        .iter()
        .map(|pairs| {
            let vars: HashMap<String, String> = pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            templater.instantiate(&template, &vars).map(|r| r.content)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(results)
}
