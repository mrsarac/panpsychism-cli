---
id: api-001
title: "RESTful API Design"
category: architecture
tags:
  - api
  - rest
  - http
  - design
privacy_tier: public
version: "1.2.0"
author: panpsychism
---

# RESTful API Design Guide

Design clean, consistent, and intuitive REST APIs.

## URL Structure

- Use nouns for resources: `/users`, `/orders`
- Use HTTP methods for actions: GET, POST, PUT, DELETE
- Nest resources logically: `/users/{id}/orders`

## Response Format

```json
{
  "data": {
    "id": "123",
    "type": "user",
    "attributes": {
      "name": "John Doe",
      "email": "john@example.com"
    }
  },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Error Handling

Return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Email is required",
    "field": "email"
  }
}
```

## Versioning

Use URL versioning for major changes: `/v1/users`, `/v2/users`
