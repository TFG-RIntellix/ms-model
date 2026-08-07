"""Global exception handlers for the FastAPI application.

Ensures consistent error responses and prevents stack trace leakage
in production.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)

async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors.

    Args:
        request: Incoming request that triggered the error.
        exc: Validation error raised by Pydantic.

    Returns:
        ``422 Unprocessable Entity`` with error details and request ID.
    """
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation failed",
            "errors": exc.errors(),
            "request_id": request_id,
        },
    )


async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions.

    Logs the full traceback but returns a generic message to the
    client so that stack traces are never exposed.

    Args:
        request: Incoming request that triggered the error.
        exc: Unhandled exception.

    Returns:
        ``500 Internal Server Error`` with a safe message and request ID.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "Unhandled exception (request_id=%s): %s: %s",
        request_id, type(exc).__name__, str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please try again later.",
            "error_type": type(exc).__name__,
            "request_id": request_id,
        },
    )


def register_exception_handlers(app: FastAPI):
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
