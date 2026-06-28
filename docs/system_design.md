# System Design — Elvy Energy ML Platform

## 1. Data Ingestion

**Sources:**
- IoT telemetry (MQTT from FerroAmp inverters, Pylontech BMS, FlowerHub gateways)
- Nord Pool Intraday spot prices (REST API, hourly)
- CRM / property records (building area, insulation, household size)

**Pattern:** MQTT → Kafka topic → consumer → BigQuery raw table  
**Local equivalent:** synthetic Parquet files in `/data/sample/`

**Freshness contract:** telemetry expected every 60s; price feed every 60min.  
Validation checks freshness threshold before training trigger.

---

## 2. Feature Store Concept

No managed feature store in v1, but the design mirrors one:

- `feature_engineering.py` = **transformation layer** (reproducible, versioned)
- Parquet in `/data/processed/` = **offline feature store**
- At inference time, the API recomputes features from the incoming request payload

**Production extension:** Feast or Tecton would decouple feature computation from  
training and serving, enabling point-in-time correct features and preventing  
train/serve skew.

---

## 3. Model Training

1. Prefect triggers training flow (cron or drift alert)
2. Data loaded from feature store
3. XGBoost trained with 5-fold CV
4. All params, metrics, feature importance logged to MLflow
5. Model artifact saved to MLflow artifact store (local or GCS)
6. If metrics pass threshold: promoted to Staging in registry

---

## 4. Model Registry

MLflow Model Registry provides:
- Named model versions (`energy-consumption-xgb`)
- Stage lifecycle: `None → Staging → Production → Archived`
- API loads `models:/energy-consumption-xgb/latest` at startup
- Rollback = promoting previous version back to Production

---

## 5. Batch Inference

```
Prefect schedule (daily/hourly)
  → load X_test or new household batch
  → load Production model from registry
  → predict consumption for next 24h
  → write predictions to BigQuery / Postgres
  → pricing engine reads predictions → generates quotes
```

---

## 6. Real-Time Inference

```
HTTP POST /predict-consumption
  → Pydantic validates input schema
  → model.predict() (loaded once at startup, in memory)
  → returns predicted kWh in <50ms
```

Model is loaded once via `lifespan` context manager — not per request.  
No cold start on inference.

---

## 7. Orchestration

Prefect flows handle:
- Data generation / ingestion
- Preprocessing
- Training + evaluation
- Model registration
- Batch prediction
- Drift monitoring

Each task is atomic, retryable (retries=1), and observable via Prefect UI.

---

## 8. Serving

FastAPI chosen for:
- Async support (handles concurrent pricing requests)
- Automatic OpenAPI/Swagger UI at `/docs`
- Pydantic schema validation — bad inputs fail fast at boundary
- Lightweight: runs in a 256MB container

---

## 9. Monitoring

**Data drift:** PSI per feature, checked on every new batch  
**Prediction drift:** distribution shift in `predicted_consumption_kwh`  
**Business metrics:** subscription quote accuracy vs actuals (lagged)  
**Infrastructure:** Prometheus endpoint (placeholder) → Grafana dashboard

Retraining trigger conditions:
- PSI > 0.25 on any top-5 feature
- MAPE > 8% on rolling 7-day evaluation window
- Manual trigger via Prefect UI

---

## 10. Scaling

| Dimension | Current | Production |
|---|---|---|
| Data volume | 50 households, SQLite | 10,000+ households, BigQuery |
| API throughput | Single instance | K8s HPA, 2–10 replicas |
| Training | Local, minutes | Cloud Run / Vertex AI, GPU optional |
| Feature store | Parquet files | Feast on GCS + BigQuery |
| Model registry | Local MLflow | MLflow on Cloud SQL |

---

## 11. Kubernetes Deployment

```
Namespace: energy-platform
├── energy-api         (Deployment, 2 replicas, LoadBalancer)
│   └── /health liveness + readiness probes
├── mlflow             (Deployment, 1 replica, ClusterIP)
└── prefect-worker     (Deployment, 1 replica)
```

Rolling update strategy: `maxUnavailable: 0` — zero downtime deploys.  
Resource limits set on all pods to prevent noisy-neighbour OOM kills.

---

## 12. CI/CD Pipeline

```
git push → GitHub Actions
  1. pytest (23 tests, <2s)
  2. docker build (API image)
  3. docker smoke test (/health)
  4. [on merge to main] kubectl apply k8s manifests
  5. Verify rollout: kubectl rollout status
  6. Slack notification
```

Model retraining is a separate CD pipeline triggered by drift alerts,  
not by code pushes — keeping ML lifecycle decoupled from service deployment.
