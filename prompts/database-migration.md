---
id: db-001
title: "PostgreSQL Migration Guide"
description: "Güvenli veritabanı migration stratejileri"
category: database
tags:
  - postgresql
  - migration
  - database
  - flyway
privacy_tier: public
version: "1.2.0"
author: mustafa
---

# PostgreSQL Migration Guide

Production-safe veritabanı migration stratejileri.

## Altın Kurallar

1. **Backwards Compatible** - Yeni schema eski kod ile çalışmalı
2. **Reversible** - Her migration'ın rollback scripti olmalı
3. **Atomic** - Transaction içinde çalışmalı
4. **Tested** - Staging'de test edilmeli

## Migration Patterns

### Column Ekleme (Safe)
```sql
ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT false;
```

### Column Silme (3 Aşama)
```sql
-- Phase 1: Kod'dan column kullanımını kaldır
-- Phase 2: Column'u nullable yap
ALTER TABLE users ALTER COLUMN old_field DROP NOT NULL;
-- Phase 3: Bir sonraki release'de column'u sil
ALTER TABLE users DROP COLUMN old_field;
```

### Index Ekleme (Concurrent)
```sql
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
```

## Rollback Script Örneği

```sql
-- Up
ALTER TABLE users ADD COLUMN status VARCHAR(20) DEFAULT 'active';

-- Down
ALTER TABLE users DROP COLUMN status;
```

## Kullanım

```
Bu migration için güvenli bir strateji öner:
[MIGRATION REQUIREMENT]
```
