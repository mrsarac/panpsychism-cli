---
id: git-001
title: "Git Commit Message Generator"
description: "Conventional Commits formatında commit mesajı oluşturma"
category: workflow
tags:
  - git
  - commit
  - conventional-commits
privacy_tier: public
version: "1.0.0"
author: mustafa
---

# Git Commit Message Generator

Conventional Commits formatında tutarlı commit mesajları oluştur.

## Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

## Types

- **feat**: Yeni özellik
- **fix**: Bug düzeltme
- **docs**: Sadece dokümantasyon
- **style**: Kod formatı (white-space, formatting)
- **refactor**: Ne bug ne feature olan kod değişikliği
- **perf**: Performans iyileştirmesi
- **test**: Test ekleme/düzeltme
- **chore**: Build process, auxiliary tools

## Örnekler

```
feat(auth): add OAuth2 PKCE support

fix(api): resolve race condition in token refresh

docs(readme): update installation instructions

refactor(core): extract validation logic to separate module
```

## Kullanım

```
Bu değişiklikler için conventional commits formatında commit mesajı öner:
[GIT DIFF]
```
