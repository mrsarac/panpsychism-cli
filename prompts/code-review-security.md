---
id: review-001
title: "Security Code Review Checklist"
description: "OWASP Top 10 odaklı güvenlik kod incelemesi"
category: security
tags:
  - security
  - review
  - owasp
  - code-review
privacy_tier: public
version: "1.0.0"
author: mustafa
---

# Security Code Review Checklist

Bu prompt, kod review sırasında güvenlik açıklarını tespit etmek için kullanılır.

## Kontrol Listesi

1. **Input Validation** - Kullanıcı girdileri sanitize ediliyor mu?
2. **SQL Injection** - Parametreli sorgular kullanılıyor mu?
3. **XSS** - Output encoding yapılıyor mu?
4. **CSRF** - Token kontrolü var mı?
5. **Authentication** - Güvenli session yönetimi var mı?
6. **Authorization** - Yetki kontrolleri yapılıyor mu?
7. **Secrets** - Hardcoded credentials var mı?
8. **Dependencies** - Güvenlik açığı olan bağımlılık var mı?

## Kullanım

```
Bu kodu OWASP Top 10 güvenlik standartlarına göre incele:
[KOD]
```
