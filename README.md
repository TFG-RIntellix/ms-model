# Credit Risk Engine Microservice

A production-grade Python microservice using **FastAPI** for credit risk assessment. Demonstrates senior-level best practices including clean architecture, dependency injection, and explainable AI (SHAP).

## Project Overview

This microservice provides:
- **Loan Risk Prediction**: Probability of Default (PD) estimation for loan applications
- **Explainability**: SHAP-based feature importance (Top 5 most impactful factors)
- **Multi-Model Support**: Extensible architecture for credit card and other product types
- **Production Ready**: Async operations, global error handling, comprehensive testing

## Architecture

```
/app
├── api/                # FastAPI routers
├── core/               # Configuration, lifespan, logging
├── schemas/            # Pydantic v2 validation models
├── services/           # Business logic & ML inference
├── ml_artifacts/       # XGBoost models, encoders
└── main.py            # Application entry point
/tests                 # Pytest unit & integration tests
Dockerfile             # Multi-stage Python 3.11+ build
```

## Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
# Clone / navigate to the project directory
cd ms-model

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Service

```bash
# Start the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# API will be available at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs (Swagger UI)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_endpoints.py
```

## API Endpoints

### Loan Risk Prediction

**Endpoint**: `POST /api/v1/risk/predict-loan`

**Request Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/risk/predict-loan" \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 35,
    "genero": "Mujer",
    "estado_civil": "Casado",
    "educacion": "Grado",
    "situacion_laboral": "Indefinido",
    "sector_trabajo": "Tecnologia",
    "dependientes": 2,
    "vivienda": "Propia_Hipoteca",
    "tiene_hipoteca": "Si",
    "ingresos_anuales": 45000.0,
    "tipo_prestamo": "Personal",
    "proposito": "Consolidacion_Deuda",
    "monto_prestamo": 15000.0,
    "plazo_meses": 36,
    "tasa_interes": 5.5,
    "ltv": 0.45,
    "dti": 0.35,
    "num_prestamos_previos": 1,
    "num_moras_previas": 0
  }'
```

**Response Example**:
```json
{
  "probability_of_default": 0.185,
  "risk_segment": "Medium",
  "shap_explanations": [
    {
      "feature": "Num_Moras_Previas",
      "impact": 0.12,
      "direction": "increase"
    },
    {
      "feature": "Ingresos_Anuales",
      "impact": -0.08,
      "direction": "decrease"
    },
    {
      "feature": "DTI",
      "impact": 0.06,
      "direction": "increase"
    },
    {
      "feature": "Tipo_Prestamo",
      "impact": -0.04,
      "direction": "decrease"
    },
    {
      "feature": "Tasa_Interes",
      "impact": 0.03,
      "direction": "increase"
    }
  ]
}
```

### Model Information

**Endpoint**: `GET /api/v1/risk/model-info`

**Response Example**:
```json
{
  "app_version": "1.1.0",
  "model_loaded": true,
  "boosting_rounds": 200,
  "feature_names": [
    "Edad", "Genero", "Estado_Civil", "Educacion",
    "Situacion_Laboral", "Sector_Trabajo", "Dependientes",
    "Vivienda", "Tiene_Hipoteca", "Ingresos_Anuales",
    "Tipo_Prestamo", "Proposito", "Monto_Prestamo",
    "Plazo_Meses", "Tasa_Interes", "LTV", "DTI",
    "Num_Prestamos_Previos", "Num_Moras_Previas"
  ],
  "scaler_loaded": true,
  "encoder_loaded": true,
  "model_artifact_date": "2026-04-02T19:30:00+00:00",
  "risk_thresholds": {
    "low_below": 0.15,
    "high_at_or_above": 0.35
  }
}
```

## Model Management

### Loading ML Artifacts

- XGBoost models are loaded **once at startup** via the FastAPI lifespan context manager
- SHAP TreeExplainer is initialized during app startup
- Models are stored in `app/ml_artifacts/` (.json and .pkl files)

### Supported Models

1. **Loan Model** (`loans_model.json`) - Default model for loan risk assessment
2. **Credit Card Model** (placeholder) - To be implemented

## Development Guidelines

### Dependency Injection

The service uses FastAPI's `Depends()` to inject models and services:

```python
@router.post("/api/v1/risk/predict-loan")
async def predict_loan(
    request: LoanApplicationRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    return await service.predict_loan(request)
```

### Async ML Operations

CPU-bound ML tasks (model.predict, SHAP calculations) are executed in a thread pool:

```python
result = await asyncio.to_thread(self.model.predict, X)
```

### Error Handling

- **422 Unprocessable Entity**: Pydantic validation errors
- **500 Internal Server Error**: ML inference failures
- Global exception handlers prevent stack trace leakage to clients

## Docker Deployment

```bash
# Build the image
docker build -t credit-risk-engine:latest .

# Run the container
docker run -p 8000:8000 credit-risk-engine:latest
```

## Configuration

Environment variables (via `app/core/settings.py`):

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Credit Risk Engine` | Service display name |
| `APP_VERSION` | *(auto from pyproject.toml)* | Semantic version |
| `DEBUG` | `False` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `MODEL_PATH` | `app/ml_artifacts/loans_model.json` | XGBoost model path |
| `ENCODER_PATH` | `app/ml_artifacts/encoder.pkl` | Categorical encoder path |
| `SCALER_PATH` | `app/ml_artifacts/scaler.pkl` | StandardScaler path |
| `API_V1_PREFIX` | `/api/v1` | API version prefix |
| `RISK_THRESHOLD_LOW` | `0.15` | PD below → Low risk |
| `RISK_THRESHOLD_HIGH` | `0.35` | PD at or above → High risk |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

## Performance Notes

- **First request latency**: Higher (model loading only on startup)
- **Subsequent requests**: <100ms (inference only)
- **SHAP calculations**: ~50-200ms depending on feature set

## License

TFG Project - Universidad

## Author

Backend Engineer & MLOps Specialist


### TODO:

+ ~~Improve how we inject the configuration into this microservice.~~ ✅ (v1.1.0)
+ ~~Improve the functionality of the API.~~ ✅ (v1.1.0 — model-info endpoint, request-id in errors)
+ Add credit card risk prediction functionality.
+ Check if it's possible to define an unique endpoint and by its content we can determine which model to use, credit card or loan one, in case of that then do the proper validation of the input data for each model.
