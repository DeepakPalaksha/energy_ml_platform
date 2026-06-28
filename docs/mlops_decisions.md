# MLOps Design Decisions

## Local vs Cloud

**Chosen**: Local-first with cloud-ready architecture.

SQLite + local Parquet for dev. All components (MLflow, Prefect, Postgres, GCS)
are swappable via environment variables. This means the project runs on a laptop
without cloud accounts, while production deployment is a config change, not a
rewrite.

**When to go cloud**: When dataset size exceeds RAM, when you need team-shared
experiment tracking, or when you need managed compute for training.

---

## Batch vs Real-Time Inference

**Chosen**: Both. The API supports real-time single-prediction requests.
Prefect supports scheduled batch prediction over a test set.

**Rationale**: Subscription pricing is computed once per household per pricing
period (batch). However, the VPP dispatch decisions need near-real-time latency
(real-time). Both patterns are supported in the same codebase.

**Production extension**: Add a Kafka consumer to the API for streaming inference
from IoT telemetry — same model, different transport layer.

---

## Simple Model vs Deep Learning

**Chosen**: XGBoost baseline first.

XGBoost wins on tabular time-series data with <1M rows, handles mixed feature types
natively, trains in seconds, and is interpretable. TFT/Chronos-2 placeholders are
noted in the architecture but not implemented in v1.

**When to upgrade**: When XGBoost MAPE plateaus and you've verified the accuracy
improvement from a TFT justifies the operational complexity (GPU serving, longer
training time, harder debugging).

---

## SQLite vs Data Warehouse

**Chosen**: SQLite for local dev, BigQuery/Postgres path documented.

SQLite requires zero setup and is sufficient for 50 households × 6 months (~200K rows).
For 10,000 households × 5 years, that's ~400M rows — BigQuery or TimescaleDB is required.
The DATABASE_URL env var makes this a one-line change.

---

## Docker Compose vs Kubernetes

**Chosen**: Docker Compose for local dev, Kubernetes manifests for production.

Docker Compose is sufficient for a single developer running all services locally.
Kubernetes adds: horizontal scaling, rolling deployments, health checks, resource
management, and self-healing. The K8s manifests in `/infra/k8s/` are ready to apply
to any cluster.

**Interview answer**: "I use Docker Compose to develop and Kubernetes to deploy.
They're not competing — they serve different contexts."

---

## Prefect vs Airflow

**Chosen**: Prefect 2.x.

Prefect flows are Python functions with decorators — no YAML, no custom operators,
no Celery/Redis required. A bioinformatician can write a Prefect flow without
platform engineering support. Airflow has better enterprise ecosystem support
(Cloud Composer, MWAA) and is the right choice for large, established data orgs.

For a 10–20 person team building new infrastructure, Prefect's lower overhead wins.

---

## MLflow Registry vs Custom Registry

**Chosen**: MLflow Model Registry.

Built-in staging lifecycle (None → Staging → Production → Archived), automatic
artifact linking, and API-compatible loading (`models:/name/latest`). Custom
registries add complexity without benefit at this scale. At hyperscale, Databricks
Unity Catalog or SageMaker Model Registry provide additional governance features.
