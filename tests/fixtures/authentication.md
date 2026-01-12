---
id: auth-001
title: "Authentication Flow Guide"
category: security
tags:
  - authentication
  - oauth
  - jwt
  - security
privacy_tier: public
version: "1.0.0"
author: panpsychism
---

# Authentication Flow Guide

This prompt helps implement secure authentication flows in applications.

## Overview

Authentication is the process of verifying user identity. This guide covers:

- OAuth 2.0 implementation
- JWT token management
- Session handling
- Password hashing with bcrypt

## Best Practices

1. Always use HTTPS for authentication endpoints
2. Implement rate limiting on login attempts
3. Use secure, httpOnly cookies for session tokens
4. Hash passwords with bcrypt (cost factor >= 12)

## Example Implementation

```typescript
import { hash, compare } from 'bcrypt';
import jwt from 'jsonwebtoken';

async function authenticate(email: string, password: string) {
  const user = await findUserByEmail(email);
  if (!user) throw new Error('User not found');

  const valid = await compare(password, user.passwordHash);
  if (!valid) throw new Error('Invalid credentials');

  return jwt.sign({ userId: user.id }, process.env.JWT_SECRET);
}
```
