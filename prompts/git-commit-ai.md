---
id: git-002
title: "AI Commit Message Generator"
category: devops
tags:
  - git
  - commit
  - automation
  - conventional-commits
privacy_tier: public
version: "1.0.0"
---

# AI Commit Message Generator

Git diff'ten anlamlı commit message üret.

## Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types
- feat: Yeni özellik
- fix: Bug fix
- docs: Dokümantasyon
- style: Formatting
- refactor: Kod refactoring
- test: Test ekleme
- chore: Bakım işleri

## Örnekler
```
feat(auth): add OAuth2 login support

- Implement Google OAuth flow
- Add token refresh mechanism
- Store tokens securely in keychain

Closes #123
```
