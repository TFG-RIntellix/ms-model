# ms-model

**Machine-learning scoring microservice for the RIntellix credit-risk platform.**

`Python 3.11+` · `FastAPI` · `XGBoost` · `SHAP` · `Layered Architecture`

---

## 1. Overview

`ms-model` is the machine-learning core of RIntellix. Given the structured attributes of a loan
or credit-card application, it returns a **probability of default (PD)** prediction together
with an **explainability breakdown** (the top contributing risk factors, via SHAP), so a risk
analyst can understand *why* the model produced a given score — not just the score itself.

It is a stateless prediction service: it does not persist data or call other RIntellix services;
it is called synchronously by `ms-risk-engine` for each simulation.

## 2. Key aspects of the system

- **Clean, layered architecture.** `api/` (FastAPI routers), `core/` (settings, app lifespan,
  logging), `schemas/` (Pydantic v2 request/response validation), `services/` (business logic
  and ML inference), `ml_artifacts/` (the trained XGBoost model and its encoders).
- **Explainable AI by design.** Every prediction is accompanied by a SHAP-based ranking of the
  top 5 most influential features, computed by `InferenceService` — this is the data source for
  the "risk drivers" section shown in the frontend and in the generated PDF report.
- **Multi-product support.** The service supports more than one credit product (personal loans
  and credit cards) through separate, explicitly-typed request schemas rather than a single
  generic payload, keeping validation strict per product.
- **Async, production-oriented FastAPI setup.** Async endpoint handlers, a documented health
  check (`/health`, used by the Docker `HEALTHCHECK`), and global error handling.
- **Reproducible dependency pinning.** All runtime dependencies are pinned to exact versions in
  `pyproject.toml` to keep model inference numerically reproducible across environments.

### Main REST endpoints

All endpoints are served under the configured API v1 prefix (`{API_V1_PREFIX}/risk`):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/risk/model-info` | Metadata about the loaded ML model and service configuration |
| `POST` | `/risk/predict-loan` | PD prediction + SHAP explanation for a loan application |
| `POST` | `/risk/predict-credit-card` | PD prediction + SHAP explanation for a credit-card application |

Interactive OpenAPI/Swagger documentation is available at `/docs` once the service is running.

### Repository structure

The following schematic illustrates the source code layout and how the key architectural pieces described above map to the main project folders:

![Directory structure](./estructura_directorios_ms_model.svg)

## 3. Tech stack

- **Language / runtime:** Python 3.11+
- **Framework:** FastAPI + Uvicorn (ASGI)
- **Validation:** Pydantic v2 / `pydantic-settings`
- **ML:** XGBoost (model), SHAP (explainability), scikit-learn, NumPy, pandas
- **Testing / tooling:** pytest, pytest-asyncio, httpx, ruff, mypy

## 4. Prerequisites

- Python 3.11 or higher
- `pip` (or `conda`)

## 5. Getting started

> `**IMPORTANT**`

> **Global platform deployment**
> This repository contains only the ML engine code. To spin up the entire RIntellix platform (including this engine, databases, and the rest of the microservices), clone the main infrastructure repository **[TFG-RIntellix/rintellix-deployment]** and follow its instructions.

The following commands are provided for local development, code review, and testing:

```bash
# 1. Clone the repository
git clone https://github.com/TFG-RIntellix/ms-model.git
cd ms-model

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install the project (runtime deps) — add "[dev]" for tests/notebooks
pip install -e ".[dev]"
```

The trained model artifacts already ship inside `app/ml_artifacts/`, so no separate training step is required.

## 6. Configuration

Runtime configuration is managed through `pydantic-settings` in `app/core/`. Override any setting via environment variables as needed for your environment (e.g. `.env` file or container environment variables).

| Variable | Description | Default |
|---|---|---|
| `API_V1_PREFIX` | Prefix for API endpoints | `/api/v1` |
| `MODEL_PATH` | Path to the trained XGBoost model | `app/ml_artifacts/model.xgb` |
| `LOG_LEVEL` | Logging detail level (INFO, DEBUG, etc.) | `INFO` |

## 7. Testing

```bash
pytest
```

Unit and integration tests live under `tests/`. Offline model-retraining scripts (not part of
the running service) live under `training/`.

## 8. Related services

- **ms-risk-engine** — the only consumer of this service; calls `/risk/predict-loan` and
  `/risk/predict-credit-card` synchronously as part of the simulation flow.

## 9. Author

Lucía Fernández Mancebo — TFG *RIntellix*, Universidad de Cantabria.



