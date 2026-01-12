---
id: auth-001
title: "OAuth2 Authorization Code Flow"
description: "PKCE ile güvenli OAuth2 implementasyonu rehberi"
category: security
tags:
  - oauth2
  - authentication
  - pkce
  - jwt
privacy_tier: public
version: "2.1.0"
author: mustafa
---

# OAuth2 Authorization Code Flow

Bu prompt, OAuth2 authorization code flow'un PKCE ile güvenli implementasyonunu kapsar.

## Adımlar

1. **Authorization Request** - Client ID, redirect URI, scope, state, code_challenge
2. **PKCE Code Verifier** - 43-128 karakter random string
3. **Code Challenge** - SHA256(code_verifier) base64url encoded
4. **Token Exchange** - Authorization code + code_verifier ile token al
5. **Token Storage** - Secure cookie veya memory-only

## Örnek Flow

```typescript
// 1. Generate PKCE
const codeVerifier = generateRandomString(64);
const codeChallenge = base64url(sha256(codeVerifier));

// 2. Authorization URL
const authUrl = `${authServer}/authorize?
  response_type=code&
  client_id=${clientId}&
  redirect_uri=${redirectUri}&
  scope=openid profile&
  state=${state}&
  code_challenge=${codeChallenge}&
  code_challenge_method=S256`;

// 3. Exchange code for tokens
const tokens = await fetch(`${authServer}/token`, {
  method: 'POST',
  body: new URLSearchParams({
    grant_type: 'authorization_code',
    code: authCode,
    redirect_uri: redirectUri,
    client_id: clientId,
    code_verifier: codeVerifier
  })
});
```

## Güvenlik Notları

- State parametresi CSRF koruması için zorunlu
- PKCE, authorization code interception saldırılarını önler
- Refresh token'ları httpOnly cookie'de sakla
