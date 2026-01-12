---
id: k8s-001
title: "Kubernetes Network Policy Guide"
description: "Pod'lar arası network izolasyonu ve güvenlik"
category: infrastructure
tags:
  - kubernetes
  - network
  - security
  - k8s
privacy_tier: public
version: "1.0.0"
author: mustafa
---

# Kubernetes Network Policy Guide

Pod'lar arası network trafiğini kontrol etme rehberi.

## Temel Kavramlar

- **Ingress**: Pod'a gelen trafik
- **Egress**: Pod'dan çıkan trafik
- **Default Deny**: Tüm trafiği engelle, sadece izin verileni aç

## Default Deny Policy

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

## Namespace Arası İzin

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-frontend
  namespace: backend
spec:
  podSelector:
    matchLabels:
      app: api
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend
    ports:
    - protocol: TCP
      port: 8080
```

## Database Erişimi

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: db-access
  namespace: database
spec:
  podSelector:
    matchLabels:
      app: postgresql
  ingress:
  - from:
    - podSelector:
        matchLabels:
          db-access: "true"
    ports:
    - protocol: TCP
      port: 5432
```

## Kullanım

```
Bu senaryoya uygun network policy oluştur:
[SENARYO]
```
