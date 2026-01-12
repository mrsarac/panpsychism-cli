# Prompt Library Example

Demonstrates the prompt management system: PromptStore, PromptSelector, and Templater.

## What This Example Does

1. **PromptStore**: Load and browse prompts from the library
2. **PromptSelector**: Select optimal prompts based on user intent
3. **Templater**: Fill in template variables dynamically
4. **TemplateRegistry**: Manage collections of templates

## The Grimoire Metaphor

- **PromptStore**: Your spellbook collection (all available prompts)
- **PromptSelector**: Finds the right spell for your intent
- **Templater**: Fills in the magical variables
- **TemplateRegistry**: Organized spellbook catalog

## Running

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Ensure prompts directory exists
ls prompts/

# Run the example
cargo run --example prompt-library
```

## Expected Output

```
=== Panpsychism Prompt Library Example ===

--- Part 1: PromptStore ---

Available prompts: 42
  - code-review (CodeReview)
  - explain-concept (Education)
  - debug-error (Debugging)
  ...

--- Part 2: PromptSelector ---

Query: I need help reviewing my authentication code
Selected 3 prompts:
  1. code-review (confidence: 0.92)
  2. security-audit (confidence: 0.85)
  3. auth-best-practices (confidence: 0.78)

--- Part 3: Templater ---

Template: code-analysis
Instantiated prompt:
Analyze the following rust code for error handling:
...

--- Part 4: Template Registry ---

Registered templates: 2
  - code-analysis
  - explain-concept
```

## Key Concepts

### PromptStore

Central storage for all prompts:

```rust
let store = PromptStore::load(&config).await?;

// Browse prompts
for prompt in store.list() {
    println!("{}: {}", prompt.id, prompt.title);
}

// Get specific prompt
let prompt = store.get("code-review")?;
```

### PromptSelector

Intelligent prompt selection:

```rust
let selector = PromptSelector::new(&store);

// Select top N prompts for a query
let selection = selector.select("review my code", 3).await?;

// Filter by category
let selection = selector
    .select_with_filter(query, 5, |p| p.category == ContentCategory::CodeReview)
    .await?;
```

### Templater

Variable substitution in templates:

```rust
let template = PromptTemplate::new("my-template", "Hello, {{name}}!");
    .with_variable(TemplateVariable::new("name", VariableType::String).required());

let mut vars = HashMap::new();
vars.insert("name".to_string(), "World".to_string());

let result = templater.instantiate(&template, &vars).await?;
// "Hello, World!"
```

### TemplateRegistry

Organized template management:

```rust
let mut registry = TemplateRegistry::new();
registry.register(template)?;

let template = registry.get("my-template")?;
```

## Template Syntax

### Variables

```
{{variable_name}}
```

### Conditionals

```
{{#if condition}}
  Content when true
{{/if}}
```

### Defaults

```rust
TemplateVariable::new("name", VariableType::String)
    .with_default("default value")
```

## Variable Types

| Type | Description |
|------|-------------|
| `String` | Short text |
| `Text` | Multi-line text |
| `Integer` | Whole numbers |
| `Float` | Decimal numbers |
| `Boolean` | true/false |
| `Array` | List of values |
| `Object` | Key-value pairs |

## Prompt File Format

Prompts are stored as YAML files:

```yaml
# prompts/code-review.yaml
id: code-review
title: Code Review Assistant
category: CodeReview
content: |
  Review the following code:

  {{code}}

  Focus on:
  - Security
  - Performance
  - Readability
variables:
  - name: code
    type: Text
    required: true
examples:
  - input: "Review this function"
    context: "fn add(a, b) { a + b }"
tags:
  - code
  - review
  - quality
```

## Next Steps

- See [basic-query](../basic-query/) for simple usage
- Explore [multi-agent](../multi-agent/) for agent workflows
- Check [llm-router](../llm-router/) for provider selection
