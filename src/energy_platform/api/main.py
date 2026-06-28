"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from energy_platform.api.routes import router
from energy_platform.config.settings import API_HOST, API_PORT
from energy_platform import __version__

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, release at shutdown."""
    logger.info("Starting up — loading model from MLflow...")
    try:
        from energy_platform.models.predict import load_model_from_latest_run
        app.state.model = load_model_from_latest_run()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load model: {e}. /predict-consumption returns 503.")
        app.state.model = None
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Elvy Energy — ML Platform API",
    description="VPP + Residential Subscription Intelligence Platform.",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("energy_platform.api.main:app", host=API_HOST, port=API_PORT, reload=True)
