"""Pydantic v2 data models for request/response validation.

This module re-exports all schemas from the split sub-modules for
backward compatibility.  New code should import directly from:

- :mod:`app.schemas.enums`
- :mod:`app.schemas.requests`
- :mod:`app.schemas.responses`
"""

# Re-export everything so existing ``from app.schemas.models import ...``
# statements continue to work.
from app.schemas.enums import (  # noqa: F401
    GeneroEnum,
    EstadoCivilEnum,
    EducacionEnum,
    SituacionLaboralEnum,
    SectorTrabajoEnum,
    ViviendaEnum,
    TieneHipotecaEnum,
    TienePrestamoEnum,
    TipoPrestamoEnum,
    PropositoEnum,
)

from app.schemas.requests import (  # noqa: F401, E402
    LoanApplicationRequest,
    CreditCardApplicationRequest,
)

from app.schemas.responses import (  # noqa: F401, E402
    SHAPExplanation,
    PredictionResponse,
    ModelInfoResponse,
    ErrorResponse,
)
