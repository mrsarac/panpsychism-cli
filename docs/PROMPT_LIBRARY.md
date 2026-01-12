# Panpsychism Prompt Library Guide

> Creating, managing, and using prompts effectively

This guide explains how to build and maintain a high-quality prompt library for Panpsychism.

---

## Table of Contents

1. [Overview](#overview)
2. [Prompt Format](#prompt-format)
3. [Creating Prompts](#creating-prompts)
4. [Organizing Prompts](#organizing-prompts)
5. [Indexing](#indexing)
6. [Searching](#searching)
7. [Best Practices](#best-practices)
8. [Example Prompts](#example-prompts)

---

## Overview

### What is a Prompt Library?

A prompt library is a collection of reusable, searchable prompt templates that Panpsychism uses to:

- **Enhance queries**: Find relevant context for user questions
- **Improve quality**: Provide domain expertise and best practices
- **Ensure consistency**: Maintain standards across responses
- **Enable specialization**: Support different use cases

### Library Structure

```
prompts/
├── development/
│   ├── code-review.md
│   ├── debugging.md
│   └── architecture.md
├── security/
│   ├── oauth2.md
│   ├── vulnerability-scan.md
│   └── penetration-testing.md
├── writing/
│   ├── blog-post.md
│   ├── documentation.md
│   └── technical-writing.md
└── master/
    ├── meta-prompts.md
    └── system-prompts.md
```

---

## Prompt Format

### File Structure

Prompts are Markdown files with YAML frontmatter:

```markdown
---
# YAML Frontmatter (metadata)
id: unique-identifier
title: "Human-Readable Title"
description: "Optional longer description"
category: category-name
tags:
  - tag1
  - tag2
privacy_tier: public
version: "1.0.0"
author: author-name
---

# Prompt Title

Prompt content goes here...

## Sections

Organize with Markdown headers...

## Examples

Include examples...
```

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string | Unique identifier | `auth-001` |
| `title` | string | Human-readable title | `"OAuth2 Guide"` |
| `category` | string | Category name | `security` |
| `tags` | list | Searchable keywords | `[oauth, jwt]` |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | string | - | Longer description for search |
| `privacy_tier` | enum | `public` | Access control level |
| `version` | string | `1.0.0` | Semantic version |
| `author` | string | - | Author attribution |
| `created_at` | datetime | - | Creation timestamp |
| `updated_at` | datetime | - | Last update timestamp |
| `deprecated` | boolean | `false` | Mark as deprecated |
| `superseded_by` | string | - | ID of replacement prompt |

### Privacy Tiers

| Tier | Description | Use Case |
|------|-------------|----------|
| `public` | Open to all users | General prompts |
| `internal` | Organization only | Company-specific |
| `confidential` | Restricted access | Sensitive topics |
| `restricted` | Highly limited | Proprietary methods |

---

## Creating Prompts

### Step 1: Identify the Use Case

Before writing, answer:

- What problem does this prompt solve?
- Who is the target user?
- What knowledge does it encode?
- What output format is expected?

### Step 2: Choose a Template

**Expert Role Prompt:**
```markdown
---
id: expert-001
title: "Domain Expert"
category: general
tags: [expert, advisor]
---

# Domain Expert

You are a world-class expert in [DOMAIN]. You have 20+ years of experience and deep knowledge of:

- [Specialty 1]
- [Specialty 2]
- [Specialty 3]

## Your Approach

When answering questions:
1. Consider the user's expertise level
2. Provide accurate, actionable information
3. Include relevant examples
4. Cite best practices when applicable
```

**Task-Specific Prompt:**
```markdown
---
id: task-001
title: "Code Review Checklist"
category: development
tags: [code, review, quality]
---

# Code Review Checklist

Review the provided code for the following aspects:

## Security
- [ ] Input validation
- [ ] SQL injection prevention
- [ ] XSS protection

## Performance
- [ ] Algorithm efficiency
- [ ] Memory usage
- [ ] Caching opportunities

## Maintainability
- [ ] Code readability
- [ ] Documentation
- [ ] Test coverage

Provide specific, actionable feedback for each issue found.
```

**Multi-Step Prompt:**
```markdown
---
id: process-001
title: "Problem-Solving Process"
category: methodology
tags: [process, problem-solving]
---

# Problem-Solving Process

Follow this structured approach:

## Step 1: Understand
- Restate the problem in your own words
- Identify constraints and requirements
- Ask clarifying questions if needed

## Step 2: Plan
- Break down into sub-problems
- Consider multiple approaches
- Select the best strategy

## Step 3: Execute
- Implement the solution step by step
- Validate each step before proceeding
- Document decisions and trade-offs

## Step 4: Verify
- Test against requirements
- Consider edge cases
- Summarize the solution
```

### Step 3: Write Content

Guidelines for prompt content:

1. **Be Specific**: Avoid vague instructions
2. **Use Examples**: Show expected input/output
3. **Structure Clearly**: Use headers, lists, code blocks
4. **Define Constraints**: Specify limits and boundaries
5. **Include Fallbacks**: Handle edge cases

### Step 4: Add Metadata

```yaml
---
id: code-review-security
title: "Security Code Review"
description: "OWASP Top 10 focused security review checklist"
category: security
tags:
  - security
  - code-review
  - owasp
  - vulnerability
privacy_tier: public
version: "1.0.0"
author: mustafa
---
```

---

## Organizing Prompts

### Category Structure

Recommended categories:

| Category | Description | Examples |
|----------|-------------|----------|
| `development` | Coding and software | Code review, debugging |
| `security` | Security practices | Auth, vulnerability |
| `writing` | Content creation | Docs, blogs, emails |
| `analysis` | Data and research | Reports, summaries |
| `business` | Strategy and planning | Plans, communications |
| `creative` | Design and ideation | Brainstorming, design |
| `devops` | Infrastructure | Deployment, monitoring |
| `master` | System prompts | Meta-prompts |

### Tagging Strategy

Use consistent, meaningful tags:

```yaml
# Good tags
tags:
  - authentication    # Specific topic
  - oauth2           # Technology
  - security         # Domain
  - best-practices   # Type
  - api              # Context

# Avoid
tags:
  - important        # Too vague
  - new              # Temporal
  - misc             # Catch-all
```

### Naming Conventions

```
prompts/
├── {category}/
│   └── {descriptive-name}.md

# Examples
prompts/security/oauth2-implementation.md
prompts/development/code-review-rust.md
prompts/writing/technical-documentation.md
```

### Versioning

Use semantic versioning in prompts:

```yaml
version: "1.0.0"   # Major.Minor.Patch
```

- **Major**: Breaking changes to prompt structure
- **Minor**: New sections or capabilities
- **Patch**: Typos, clarifications, small fixes

---

## Indexing

### Basic Indexing

```bash
# Index default directory
panpsychism index

# Output:
# Indexing complete!
#    Indexed: 42 prompts
#    Output: ./data/masters.mv2
```

### Custom Indexing

```bash
# Custom directories
panpsychism index --dir /path/to/prompts --output ./data/library.mv2

# Force re-index
panpsychism index --force

# Verbose progress
panpsychism index --verbose
```

### Index File Format

The `.mv2` file is a binary index containing:

- Prompt metadata (id, title, tags)
- TF-IDF vectors for search
- Category and tag indices
- Full text for synthesis

### When to Re-Index

Re-index when you:

- Add new prompts
- Modify existing prompts
- Delete prompts
- Change file structure

---

## Searching

### Basic Search

```bash
panpsychism search "authentication"
```

### Advanced Search

```bash
# Filter by category
panpsychism search "api" --category development

# Filter by tags
panpsychism search "review" --tags "security,code"

# More results
panpsychism search "kubernetes" --top 10

# Lower threshold
panpsychism search "database" --min-score 0.01
```

### Search Scoring

Results are scored using TF-IDF (Term Frequency-Inverse Document Frequency):

| Score | Meaning |
|-------|---------|
| 0.90+ | Excellent match |
| 0.70-0.89 | Good match |
| 0.50-0.69 | Moderate match |
| 0.30-0.49 | Weak match |
| <0.30 | Poor match |

### Search Tips

1. **Use keywords**: "oauth2 authentication" > "how to do auth"
2. **Be specific**: "rust memory safety" > "code review"
3. **Combine tags**: `--tags "security,api"`
4. **Check categories**: Use `--category` to narrow down

---

## Best Practices

### Prompt Quality

1. **Clear Purpose**: Each prompt should have one clear goal
2. **Actionable Output**: Define what the response should look like
3. **Domain Knowledge**: Include expert-level information
4. **Examples**: Show input/output pairs
5. **Constraints**: Specify limits and boundaries

### Metadata Quality

1. **Unique IDs**: Use descriptive, unique identifiers
2. **Accurate Tags**: Tags should reflect content
3. **Proper Categories**: Place in correct category
4. **Version Tracking**: Update version on changes

### Maintenance

1. **Regular Review**: Audit prompts quarterly
2. **Deprecation**: Mark outdated prompts as deprecated
3. **Documentation**: Keep README in prompts directory
4. **Testing**: Verify prompts produce expected results

### Anti-Patterns to Avoid

```markdown
# BAD: Too vague
You are helpful. Answer questions.

# GOOD: Specific role and instructions
You are a senior Rust developer with expertise in memory safety.
When reviewing code, focus on ownership, borrowing, and lifetime issues.
Provide specific line references and suggest fixes.
```

```markdown
# BAD: No structure
Just do code review and find bugs and security issues and also
check performance and make sure it follows best practices.

# GOOD: Structured checklist
## Review Checklist

### Security
- [ ] Input validation
- [ ] SQL injection prevention

### Performance
- [ ] Algorithm complexity
- [ ] Memory allocation
```

---

## Example Prompts

### Development: Code Review

```markdown
---
id: dev-code-review-001
title: "Comprehensive Code Review"
description: "Multi-aspect code review covering security, performance, and maintainability"
category: development
tags:
  - code-review
  - quality
  - best-practices
privacy_tier: public
version: "1.0.0"
---

# Comprehensive Code Review

You are a senior software engineer conducting a code review.

## Review Criteria

### 1. Security (Critical)
- Input validation and sanitization
- Authentication and authorization checks
- Injection vulnerability prevention (SQL, XSS, etc.)
- Secrets management (no hardcoded credentials)
- Secure communication (TLS, encryption)

### 2. Correctness (High)
- Logic errors and edge cases
- Error handling and recovery
- Null/undefined safety
- Race conditions and concurrency issues

### 3. Performance (Medium)
- Algorithm complexity (time/space)
- Database query optimization
- Memory management
- Caching opportunities

### 4. Maintainability (Medium)
- Code readability and clarity
- Naming conventions
- Documentation and comments
- Test coverage

## Output Format

For each issue found:
1. **Location**: File and line number
2. **Severity**: Critical / High / Medium / Low
3. **Category**: Security / Correctness / Performance / Maintainability
4. **Description**: What's wrong
5. **Suggestion**: How to fix it

## Example

**Location**: `src/auth.rs:45`
**Severity**: Critical
**Category**: Security
**Description**: Password is compared using `==` which is vulnerable to timing attacks.
**Suggestion**: Use `constant_time_eq` from the `subtle` crate for secure comparison.
```

### Security: OAuth2 Implementation

```markdown
---
id: sec-oauth2-001
title: "OAuth2 Implementation Guide"
description: "Step-by-step guide for implementing OAuth2 authentication"
category: security
tags:
  - oauth2
  - authentication
  - authorization
  - jwt
privacy_tier: public
version: "1.0.0"
---

# OAuth2 Implementation Guide

## Overview

This guide covers implementing OAuth2 Authorization Code flow with PKCE.

## Prerequisites

- OAuth provider credentials (Client ID, Client Secret)
- HTTPS endpoint for callback URL
- Secure token storage mechanism

## Implementation Steps

### 1. Generate PKCE Parameters

```rust
use sha2::{Sha256, Digest};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};

fn generate_pkce() -> (String, String) {
    let verifier: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(64)
        .map(char::from)
        .collect();

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(hasher.finalize());

    (verifier, challenge)
}
```

### 2. Build Authorization URL

```rust
fn build_auth_url(client_id: &str, redirect_uri: &str, challenge: &str) -> String {
    format!(
        "https://provider.com/authorize?\
         response_type=code&\
         client_id={}&\
         redirect_uri={}&\
         code_challenge={}&\
         code_challenge_method=S256&\
         scope=openid%20profile%20email",
        client_id, redirect_uri, challenge
    )
}
```

### 3. Handle Callback

```rust
async fn handle_callback(code: &str, verifier: &str) -> Result<TokenResponse> {
    let response = reqwest::Client::new()
        .post("https://provider.com/token")
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("code_verifier", verifier),
            ("client_id", CLIENT_ID),
            ("redirect_uri", REDIRECT_URI),
        ])
        .send()
        .await?;

    Ok(response.json().await?)
}
```

### 4. Validate and Store Tokens

- Store refresh token securely (encrypted at rest)
- Keep access token in memory when possible
- Implement automatic refresh before expiry

## Security Considerations

- Always use PKCE for public clients
- Validate `state` parameter to prevent CSRF
- Use short-lived access tokens (15 min - 1 hour)
- Implement token revocation on logout
```

### Writing: Technical Documentation

```markdown
---
id: write-tech-docs-001
title: "Technical Documentation Writer"
description: "Create clear, comprehensive technical documentation"
category: writing
tags:
  - documentation
  - technical-writing
  - api-docs
privacy_tier: public
version: "1.0.0"
---

# Technical Documentation Writer

You are a technical writer creating documentation for developers.

## Documentation Principles

1. **Clarity**: Use simple, direct language
2. **Completeness**: Cover all use cases
3. **Examples**: Show working code
4. **Structure**: Logical organization

## Document Structure

### For API Documentation

```markdown
# Endpoint Name

Brief description of what this endpoint does.

## Request

`METHOD /path`

### Headers
| Header | Required | Description |
|--------|----------|-------------|
| Authorization | Yes | Bearer token |

### Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| id | string | Yes | Resource ID |

### Body
\`\`\`json
{
  "field": "value"
}
\`\`\`

## Response

### Success (200)
\`\`\`json
{
  "data": {...}
}
\`\`\`

### Error (400)
\`\`\`json
{
  "error": "Description"
}
\`\`\`

## Example

\`\`\`bash
curl -X POST https://api.example.com/path \
     -H "Authorization: Bearer token" \
     -d '{"field": "value"}'
\`\`\`
```

## Writing Style

- Use active voice
- Be concise but complete
- Define acronyms on first use
- Include version numbers
- Date the documentation
```

---

## Summary

A well-organized prompt library is the foundation of effective Panpsychism usage:

1. **Create**: Write prompts with clear purpose and structure
2. **Organize**: Use categories and tags consistently
3. **Index**: Keep the index updated
4. **Search**: Use keywords and filters effectively
5. **Maintain**: Review and update prompts regularly

---

*For CLI usage, see [CLI Reference](CLI_REFERENCE.md). For system architecture, see [Architecture](ARCHITECTURE.md).*
