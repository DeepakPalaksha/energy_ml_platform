"""Central configuration — loaded from environment / .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODEL_DIR = ROOT_DIR / "models"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "energy-consumption-forecast"
)

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
ENV: str = os.getenv("ENV", "development")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/energy_platform.db")

# ── Model training ────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# ── Subscription pricing constants ────────────────────────────────────────────
RISK_BUFFER_PCT: float = 0.12        # 12% risk buffer on top of expected cost
SERVICE_MARGIN_PCT: float = 0.18     # 18% service margin
HARDWARE_AMORTIZATION_MONTHLY: float = 45.0   # EUR per household per month

# ── VPP battery defaults ──────────────────────────────────────────────────────
DEFAULT_BATTERY_CAPACITY_KWH: float = 10.0
DEFAULT_MAX_CHARGE_RATE_KW: float = 5.0
DEFAULT_EFFICIENCY: float = 0.92

# ── Data generation ───────────────────────────────────────────────────────────
N_HOUSEHOLDS: int = 50
N_MONTHS: int = 6
