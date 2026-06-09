"""FastAPI application entry point.

Initialises the Credit Risk Engine microservice with application-wide
settings, exception handlers, CORS, X-Request-ID tracing middleware,
and routers.

Typical usage::

    uvicorn app.main:app --reload
"""

import logging
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.core.settings import get_settings, configure_logging
from app.core.lifespan import lifespan
from app.core.exceptions import register_exception_handlers
from app.api.risk_router import router as risk_router

# Configure logging early
settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# OpenAPI tag metadata
# ------------------------------------------------------------------
OPENAPI_TAGS = [
    {
        "name": "Risk Assessment",
        "description": "Loan and credit card risk prediction endpoints.",
    },
    {
        "name": "Health",
        "description": "Service health and readiness probes.",
    },
    {
        "name": "Info",
        "description": "General API information.",
    },
]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Fully configured ``FastAPI`` application instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-grade microservice for credit risk assessment "
            "using XGBoost and SHAP."
        ),
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
    )

    # ------------------------------------------------------------------
    # Global exception handlers
    # ------------------------------------------------------------------
    register_exception_handlers(app)

    # ------------------------------------------------------------------
    # X-Request-ID middleware for log correlation
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        """Inject a unique ``X-Request-ID`` header for log correlation.

        If the incoming request already carries an ``X-Request-ID`` it is
        reused; otherwise a new UUID4 is generated.  The ID is added to
        both request state and the response headers.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware / route handler.

        Returns:
            ``Response`` with ``X-Request-ID`` header set.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ------------------------------------------------------------------
    # CORS — origins come from Settings; credentials only when
    #         specific origins are configured (never with wildcard).
    # ------------------------------------------------------------------
    _origins = settings.CORS_ORIGINS
    _allow_credentials = _origins != ["*"] and len(_origins) > 0
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=_allow_credentials,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(risk_router)

    # ------------------------------------------------------------------
    # Utility endpoints
    # ------------------------------------------------------------------

    @app.get("/health", tags=["Health"], summary="Health Check")
    async def health_check():
        """Return service health including model-loaded and scaler status."""
        model_manager = getattr(app.state, "model_manager", None)
        model_ready = model_manager is not None and model_manager.model_loaded
        scaler_ready = model_manager is not None and model_manager.scaler_loaded

        return {
            "status": "healthy" if model_ready else "degraded",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "model_loaded": model_ready,
            "scaler_loaded": scaler_ready,
        }

    @app.get("/", tags=["Info"], summary="API Information")
    async def root():
        """Return API information and documentation links."""
        info: dict = {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
        }
        if settings.DEBUG:
            info["docs"] = "/docs"
            info["openapi_schema"] = "/openapi.json"
        return info

    logger.info(
        "FastAPI application created: %s v%s",
        settings.APP_NAME, settings.APP_VERSION,
    )
    return app


# Application instance used by uvicorn / gunicorn
app = create_app()
