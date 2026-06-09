"""Unit tests for app/core/exceptions.py.

Covers the general_exception_handler path (coverage was 82 %).
"""

import json
from unittest.mock import Mock

import pytest

from app.core.exceptions import general_exception_handler


@pytest.mark.asyncio
async def test_general_exception_handler_returns_500():
    """Unhandled exceptions return 500 with safe message and request_id."""
    request = Mock()
    request.state.request_id = "test-req-id-123"

    exc = RuntimeError("something went wrong")
    response = await general_exception_handler(request, exc)

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error_type"] == "RuntimeError"
    assert body["request_id"] == "test-req-id-123"
    assert "internal server error" in body["detail"].lower()


@pytest.mark.asyncio
async def test_general_exception_handler_no_request_id():
    """Handler works when request.state has no request_id attribute."""
    request = Mock()
    request.state = Mock(spec=[])  # no request_id attr

    exc = ValueError("bad")
    response = await general_exception_handler(request, exc)

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["request_id"] is None
    assert body["error_type"] == "ValueError"
