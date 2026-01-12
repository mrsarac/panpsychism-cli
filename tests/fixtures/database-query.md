---
id: db-001
title: "Database Query Optimization"
category: backend
tags:
  - database
  - postgresql
  - optimization
  - sql
privacy_tier: internal
version: "2.1.0"
author: panpsychism
---

# Database Query Optimization

Optimize your SQL queries for better performance.

## Key Principles

- Use indexes on frequently queried columns
- Avoid SELECT * in production code
- Use EXPLAIN ANALYZE to understand query plans
- Batch inserts for bulk operations

## Index Strategy

```sql
-- Create index for common lookups
CREATE INDEX idx_users_email ON users(email);

-- Composite index for range queries
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at DESC);

-- Partial index for filtered queries
CREATE INDEX idx_active_users ON users(id) WHERE status = 'active';
```

## Query Examples

```sql
-- Efficient pagination
SELECT id, name, email
FROM users
WHERE id > :last_id
ORDER BY id
LIMIT 20;

-- Avoiding N+1 with JOIN
SELECT u.id, u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.id;
```
