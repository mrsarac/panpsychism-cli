---
id: rust-001
title: "Rust Error Handling Best Practices"
category: programming
tags:
  - rust
  - error-handling
  - thiserror
  - anyhow
privacy_tier: public
version: "1.0.0"
---

# Rust Error Handling

Rust'ta hata yönetimi için best practices.

## thiserror ile Custom Errors
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Not found: {0}")]
    NotFound(String),
}
```

## anyhow ile Hızlı Prototyping
```rust
use anyhow::{Context, Result};

fn read_config() -> Result<Config> {
    let content = std::fs::read_to_string("config.toml")
        .context("Failed to read config file")?;
    Ok(toml::from_str(&content)?)
}
```
