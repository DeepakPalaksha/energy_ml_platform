# Elvy Energy — VPP + Residential Subscription Intelligence Platform

**Portfolio project demonstrating production ML engineering: MLOps, FastAPI, Prefect, MLflow, Docker, Kubernetes.**

---

## Business Problem

Elvy Energy converts residential electricity billing into fixed monthly subscriptions.
To price a 15-year contract correctly, we need to:

1. **Forecast hourly consumption** per household (ML)
2. **Estimate subscription price** with risk-adjusted confidence bands (pricing engine)
3. **Simulate battery flexibility** for VPP market participation (arbitrage simulator)

A 10% forecasting error translates directly to financial loss on multi-year contracts.

---

## Architecture

```
IoT Telemetry + Spot Prices
        ↓
  Prefect Pipeline
  (generate → preprocess → train → evaluate)
        ↓
  XGBoost Model ←→ MLflow Tracking + Registry
        ↓
  FastAPI Server
  ├── /predict-consumption
  ├── /estimate-subscription
  └── /simulate-vpp
        ↓
  Kubernetes Deployment (2 replicas, HPA-ready)
```

Full detail: [`docs/architecture.md`](docs/architecture.md)

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML Model | XGBoost (baseline) |
| Experiment Tracking | MLflow |
| Orchestration | Prefect 2.x |
| API | FastAPI + Pydantic |
| Containerisation | Docker + Docker Compose |
| Deployment | Kubernetes |
| Testing | pytest |
| Monitoring | PSI drift detection (Prometheus-ready) |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker + Docker Compose

### 1. Clone and install

```bash
git clone https://github.com/your-handle/energy-ml-platform.git
cd energy-ml-platform

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
export PYTHONPATH=src
```

### 2. Generate synthetic data

```bash
python src/energy_platform/data/generate_sample_data.py
```

### 3. Preprocess + train

```bash
python -c "
import sys; sys.path.insert(0, 'src')
import pandas as pd
from energy_platform.data.preprocessing import preprocess
from energy_platform.data.generate_sample_data import generate_dataset
df = generate_dataset(save=True)
preprocess(df, save=True)
"

python src/energy_platform/models/train.py
```

### 4. Start MLflow UI

```bash
mlflow ui --port 5001
# Open http://localhost:5001
```

### 5. Start API

```bash
uvicorn energy_platform.api.main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```

### 6. Run Prefect full pipeline

```bash
prefect server start &   # In a separate terminal
python src/energy_platform/pipelines/prefect_flows.py
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
# Start API + MLflow
docker-compose up

# Start with Prefect orchestration
docker-compose --profile orchestration up

# Run training container
docker-compose --profile training run training
```

---

## Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/mlflow-deployment.yaml
kubectl apply -f infra/k8s/api-deployment.yaml

# Check status
kubectl get pods -n energy-platform
kubectl get services -n energy-platform
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Project status |
| GET | `/health` | Health check |
| POST | `/predict-consumption` | Predict hourly kWh |
| POST | `/estimate-subscription` | Monthly subscription quote |
| POST | `/simulate-vpp` | Battery arbitrage simulation |

Full schema: `http://localhost:8000/docs`

---

## Example API Call

```bash
# Subscription pricing
curl -X POST http://localhost:8000/estimate-subscription \
  -H "Content-Type: application/json" \
  -d '{
    "predicted_hourly_kwh": [0.5, 0.6, 0.4, 0.3, 0.2, 0.2, 0.3, 0.8, 1.2, 1.0,
                              0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 1.1, 1.5, 1.4, 1.3,
                              1.2, 1.0, 0.8, 0.6],
    "spot_prices_eur_mwh": [45, 42, 40, 38, 37, 36, 38, 55, 70, 65,
                             60, 58, 55, 54, 56, 60, 68, 90, 85, 80,
                             72, 65, 58, 50],
    "hardware_amortization_eur": 45.0
  }'
```

---

## Interview Talking Points

Full notes including the **NEC OncoImmunity biotech parallel**: [`docs/interview_notes.md`](docs/interview_notes.md)

MLOps design decisions and tradeoffs: [`docs/mlops_decisions.md`](docs/mlops_decisions.md)

---

## Repository Structure

```
energy-ml-platform/
├── src/energy_platform/
│   ├── config/settings.py          # Central config
│   ├── data/
│   │   ├── generate_sample_data.py # Synthetic data generator
│   │   ├── feature_engineering.py  # 27 features
│   │   └── preprocessing.py        # Validate → clean → split
│   ├── models/
│   │   ├── train.py                # XGBoost + MLflow logging
│   │   ├── evaluate.py             # MAE / RMSE / MAPE / R²
│   │   └── predict.py              # Registry-aware inference
│   ├── api/
│   │   ├── main.py                 # FastAPI app + lifespan
│   │   ├── routes.py               # All endpoints
│   │   └── schemas.py              # Pydantic models
│   ├── pricing/
│   │   └── subscription_pricing.py # Monte Carlo pricing engine
│   ├── vpp/
│   │   └── battery_simulator.py    # Charge/discharge arbitrage
│   ├── pipelines/
│   │   └── prefect_flows.py        # Orchestrated ML lifecycle
│   └── monitoring/
│       └── drift_detection.py      # PSI drift detection
├── tests/                          # pytest suite
├── infra/
│   ├── docker/                     # Dockerfiles
│   └── k8s/                        # Kubernetes manifests
├── docs/                           # Architecture + interview notes
├── docker-compose.yml
└── requirements.txt
```
