# System Architecture — Elvy Energy ML Platform

## Overview

The platform is a production-grade ML system that converts raw residential IoT telemetry
into three business outputs:

1. **Hourly consumption forecast** — used to price long-term energy subscriptions
2. **Subscription price quote** — risk-adjusted monthly fixed price per household
3. **VPP flexibility estimate** — estimated arbitrage value for battery dispatch

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                               │
│  IoT Telemetry (MQTT) │ Spot Prices (Nord Pool) │ CRM / Property  │
└────────────┬──────────────────────────┬──────────────────────────┘
             │                          │
             ▼                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE (Prefect)                      │
│  Ingestion → Validation → Cleaning → Feature Engineering          │
│  Output: Versioned Parquet splits in /data/processed/             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
    ┌──────────────┐ ┌─────────────┐ ┌─────────────┐
    │  XGBoost     │ │  MLflow     │ │  Model      │
    │  Training    │ │  Tracking   │ │  Registry   │
    └──────┬───────┘ └─────────────┘ └──────┬──────┘
           │                                 │
           └──────────────┬──────────────────┘
                          ▼
          ┌───────────────────────────────┐
          │        FastAPI Server         │
          │  /predict-consumption         │
          │  /estimate-subscription       │
          │  /simulate-vpp                │
          └───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    Subscription     VPP Dispatch     Monitoring
    Pricing Engine   Optimizer        (PSI Drift)
```

---

## Component Breakdown

### 1. Data Pipeline
- **Ingestion**: Reads from IoT telemetry + Nord Pool price feeds
- **Validation**: Schema checks, null rates, value bounds
- **Feature Engineering**: 27 features — calendar, lag, rolling, weather interaction
- **Storage**: Parquet in `/data/processed/` (local), or GCS buckets in production

### 2. Model Training
- **Algorithm**: XGBoost (baseline) — interpretable, fast, handles mixed feature types
- **Tracking**: MLflow — all hyperparameters, metrics, feature importance per run
- **Registry**: MLflow Model Registry — versioned, staged (Staging → Production)
- **Evaluation**: MAE, RMSE, MAPE, R² on held-out test set

### 3. API Serving
- **Framework**: FastAPI — async, typed, auto-generates OpenAPI docs
- **Model loading**: Loaded once at startup via `lifespan` context manager
- **Endpoints**: predict, price, simulate — fully schema-validated with Pydantic

### 4. Orchestration
- **Tool**: Prefect 2.x — DAG-free, Python-native, self-hosted or Prefect Cloud
- **Flows**: generate → preprocess → train → evaluate → (batch predict)
- **Tasks**: atomic, retryable, observable via Prefect UI

### 5. VPP Module
- Rule-based battery dispatch: charge cheap, discharge expensive
- Computes Nord Pool arbitrage value per battery configuration
- Can be extended to RL-based dispatch (SAC/MILP)

### 6. Subscription Pricing
- Monte Carlo simulation (1000 scenarios) for confidence bands
- Inputs: predicted consumption, spot prices, hardware amortization, margins
- 15-year contract pricing — financial risk is real

### 7. Monitoring
- **Drift detection**: PSI (Population Stability Index) per feature
- Thresholds: warn at 0.1, alert at 0.25
- Designed for Prometheus/Grafana integration

---

## Deployment Topology

```
Local Dev:       docker-compose up (API + MLflow + optional Prefect)
Staging/Prod:    Kubernetes (namespace: energy-platform)
                 - energy-api: 2 replicas, HPA-ready
                 - mlflow: 1 replica, ClusterIP
                 - prefect-worker: 1 replica
CI/CD:           GitHub Actions → build → test → push image → kubectl apply
```
