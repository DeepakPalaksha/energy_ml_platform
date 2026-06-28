# Interview Notes — ML Platform Engineer

## For NEC OncoImmunity (Oslo) — How to Bridge VPP → Biotech

> The ML engineering problems are identical. Only the domain data differs.
> Lead every answer with the pattern, then mention the energy implementation,
> then explicitly state how it maps to immunology/vaccine pipeline work.

### The Bridge Statement (use to open the interview)

> "My career started in biomedical signal processing — cardiac data at NTNU, clinical
> ML at Harvard MGH, and EGM feature extraction at LivaNova. I then spent 6 years
> applying those same ML engineering skills to energy systems, which gave me the chance
> to build complete ML infrastructure from scratch at scale. What I'm showing you today
> reflects that infrastructure work — and I want to explain how every design decision
> maps directly to what you'd need for an antigen prediction pipeline."

---

## Q&A — Technical Interview Prep

### Why FastAPI?

**Energy answer**: FastAPI is async-native, which handles concurrent pricing requests
from the fleet management dashboard without blocking. Auto-generated OpenAPI docs
reduce onboarding friction for the team. Pydantic validation catches bad feature
inputs at the API boundary before they reach the model.

**NEC bridge**: Same applies — your Indit pipeline integration endpoint needs schema
validation on incoming biological feature vectors. FastAPI's typed contracts mean
a bioinformatician sending wrong-shaped HLA features gets a clear error, not a
silent bad prediction.

---

### Why Prefect (not Airflow)?

**Energy answer**: Prefect 2.x is Python-native — flows are just decorated functions,
not YAML DAGs. This means the data scientist or bioinformatician can write and own
their pipeline without learning a separate DSL. Local runs and cloud deployment
use the same code. Airflow requires a Celery/Redis stack that's heavyweight for a
12-person team.

**NEC bridge**: At a 16-person biotech team, Prefect lets the immunologist-adjacent
engineer write the data prep flow without DevOps support. Airflow would require a
dedicated platform engineer just to maintain the scheduler.

**Tradeoff to know**: Airflow has better enterprise ecosystem support (Google Cloud
Composer, MWAA). If NEC scales to a 100+ person bioinformatics team, Airflow becomes
worth it. Prefect is the right call at the current stage.

---

### Why MLflow?

**Energy answer**: MLflow solves three problems in one tool: (1) experiment tracking
so I can compare Chronos-2 vs TFT vs XGBoost on the same held-out split, (2) model
registry so the API always loads the `Production`-staged model, not a random artifact,
(3) reproducibility — every run logs exact parameters, so I can re-run any experiment
6 months later.

**NEC bridge**: For antigen/T-cell activation models, you'll run hundreds of
experiments varying epitope encodings, HLA allele representations, and model
architectures. MLflow gives you a structured audit trail — which is critical when
a biologist asks "why did the model predict this peptide as immunogenic?" You need
to trace back to the exact training run.

**Alternatives**: Weights & Biases (better UI, SaaS cost), DVC (Git-centric,
good for data versioning), Neptune. MLflow is the strongest open-source choice
for self-hosted deployment.

---

### Why XGBoost as baseline?

**Energy answer**: XGBoost is interpretable via feature importance, handles missing
values natively, trains in seconds on a laptop, and outperforms neural nets on tabular
data with <1M rows. It lets me establish a reliable benchmark before introducing
complexity. The key insight: a 5% MAPE improvement from a TFT doesn't justify
3x engineering complexity unless you've first proven XGBoost can't do the job.

**NEC bridge**: Same logic for peptide-MHC binding prediction. Start with a
Random Forest or XGBoost on physicochemical features. It will be fast to train,
explainable to immunologists, and give you a performance floor. Then benchmark
against NetMHCpan or transformer-based models. The biologist should be able to
understand why the model ranked a particular peptide highly.

---

### Why Kubernetes?

**Energy answer**: The API must handle burst traffic when onboarding a new building
block (300 homes simultaneously requesting pricing). K8s gives us: horizontal pod
autoscaling, rolling deployments (zero downtime), resource limits per pod, and
a declarative infrastructure that's version-controlled alongside the code.

**NEC bridge**: For inference serving of a vaccine design pipeline — especially when
running batch predictions across thousands of peptide candidates — K8s lets you
scale the inference pods horizontally and kill them after the batch completes.
You're not paying for idle compute.

**Key talking point**: Kubernetes adds operational complexity. For a small team, a
simpler path is Cloud Run (GCP) or AWS ECS. I'd choose K8s only when you need
fine-grained resource control or multi-service orchestration.

---

### How does this scale in production?

**Current state**: Single-node SQLite + local Parquet. Handles ~50 households, dev/demo.

**Scale path**:
1. Replace SQLite with PostgreSQL (or TimescaleDB for time-series)
2. Replace local Parquet with GCS/S3 + DVC for data versioning
3. Replace single MLflow server with MLflow on Cloud SQL backend + GCS artifacts
4. Add Kubernetes HPA for API pods based on CPU/RPS
5. Introduce feature store (Feast or Tecton) to decouple feature computation from training

**NEC equivalent**: Replace household telemetry with genomic/proteomic datasets.
Replace GCS Parquet with a bioinformatics data lake (S3 + Parquet + Iceberg).
Same MLOps stack applies exactly.

---

### How do you monitor model drift?

**Energy answer**: Population Stability Index (PSI) on feature distributions.
PSI < 0.1: no action. 0.1–0.25: monitor closely. >0.25: trigger retraining pipeline.
I also monitor prediction drift — if the distribution of predicted consumption
values shifts significantly from the training distribution, that's a signal
independent of feature drift.

**NEC bridge**: For immunogenicity prediction, drift could come from:
- New HLA allele prevalence in your patient cohort
- Updated experimental assay introducing systematic bias
- New viral variants shifting epitope landscape

PSI on input features + prediction confidence scores gives you early warning.
This maps directly to your `monitoring/drift_detection.py` module.

---

### How do you design CI/CD for this?

```
Push to main branch
  → GitHub Actions triggered
  → pytest (unit + integration tests)
  → docker build (API + training images)
  → docker push to registry
  → kubectl apply -f infra/k8s/
  → smoke test /health endpoint
  → notify team (Slack)
```

**Model retraining CI**: Separate trigger — either cron (weekly) or drift alert.
Retraining runs in the training container, logs to MLflow, promotes to Staging.
Manual approval promotes to Production. This is the pattern from Elvy Energy.

---

### What are the failure points?

| Component | Failure Mode | Mitigation |
|---|---|---|
| Feature store | Stale lag features at inference time | Feature freshness checks in preprocessing |
| MLflow | Model not registered → API returns 503 | Fallback to latest run artifact |
| Spot price feed | Missing prices → bad subscription quote | Forward-fill max 2h, alert if > 2h gap |
| Battery sim | Extreme price spikes distort arbitrage | Cap spot at 99th percentile |
| K8s pod | OOM kill during batch prediction | Memory limits + chunked inference |

---

### What DevOps knowledge matters most?

1. **Docker layer caching** — put `COPY requirements.txt` before `COPY src/` so
   code changes don't invalidate the dependency layer
2. **K8s resource requests vs limits** — always set both; requests for scheduling,
   limits to prevent noisy neighbors
3. **Liveness vs readiness probes** — liveness restarts a crashed pod;
   readiness removes it from load balancer until model is loaded
4. **Rolling deployments** — `strategy: RollingUpdate` + `maxUnavailable: 0`
   for zero-downtime model updates
5. **Secret management** — never commit `.env` with real credentials; use K8s
   Secrets or GCP Secret Manager

---

## One-Paragraph Summary for "Tell me about this project"

> "This is a production-style ML platform I built to demonstrate the full engineering
> lifecycle for a residential energy subscription company. It covers everything from
> synthetic data generation and feature engineering, through XGBoost training tracked
> in MLflow, to a FastAPI serving layer and Prefect orchestration of the full pipeline.
> The business problem is real — predicting household electricity consumption for 15-year
> fixed-price contracts, where a 10% model error translates directly to financial loss.
> I also built a VPP battery simulator for Nord Pool arbitrage and a Monte Carlo
> subscription pricing engine with confidence bands. The whole thing runs locally with
> docker-compose, and I've provided K8s manifests for production deployment. Every
> design decision here — from PSI drift detection to model registry staging — maps
> directly to the ML engineering challenges at NEC OncoImmunity, just with peptide
> sequences and HLA alleles instead of kilowatt-hours."
