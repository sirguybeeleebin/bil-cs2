import json
from unittest.mock import patch

import pytest
from django.http import HttpResponse, JsonResponse
from django.test import RequestFactory

from app.middlewares.logging import LoggingMiddleware


@pytest.fixture
def get_response():
    def _view(request):
        return JsonResponse({"message": "ok"})

    return _view


@pytest.fixture
def middleware(get_response):
    return LoggingMiddleware(get_response)


@pytest.mark.django_db
def test_middleware_logs_request_and_response_json(middleware):
    factory = RequestFactory()
    request = factory.post(
        "/test-path/",
        data=json.dumps({"key": "value"}),
        content_type="application/json",
    )

    with patch("app.middlewares.logging.logger") as mock_logger:  # fixed patch target
        response = middleware(request)

    assert response.status_code == 200
    assert json.loads(response.content) == {"message": "ok"}

    assert mock_logger.info.call_count == 2

    request_log_str = mock_logger.info.call_args_list[0][0][0]
    request_log = json.loads(request_log_str)
    assert request_log["type"] == "request"
    assert request_log["method"] == "POST"
    assert request_log["path"] == "/test-path/"
    assert request_log["body"] == {"key": "value"}
    assert "request_id" in request_log

    # Check response log
    response_log_str = mock_logger.info.call_args_list[1][0][0]
    response_log = json.loads(response_log_str)
    assert response_log["type"] == "response"
    assert response_log["method"] == "POST"
    assert response_log["path"] == "/test-path/"
    assert response_log["status_code"] == 200
    assert response_log["body"] == {"message": "ok"}
    assert response_log["request_id"] == request_log["request_id"]


def test_middleware_handles_non_json_body(middleware):
    factory = RequestFactory()
    request = factory.post("/non-json/", data="not a json", content_type="text/plain")

    with patch("app.middlewares.logging.logger") as mock_logger:  # fixed patch target
        middleware(request)

    request_log_str = mock_logger.info.call_args_list[0][0][0]
    request_log = json.loads(request_log_str)
    assert request_log["body"] == "not a json"

    response_log_str = mock_logger.info.call_args_list[1][0][0]
    response_log = json.loads(response_log_str)
    assert response_log["body"] == {"message": "ok"}


def test_middleware_handles_non_json_response():
    """Middleware should handle responses that are not JSON serializable."""

    def get_response(request):
        return HttpResponse("plain text response", content_type="text/plain")

    middleware = LoggingMiddleware(get_response)
    factory = RequestFactory()
    request = factory.get("/plain-response/")

    with patch("app.middlewares.logging.logger") as mock_logger:
        middleware(request)

    response_log_str = mock_logger.info.call_args_list[1][0][0]
    response_log = json.loads(response_log_str)
    assert response_log["body"] == "plain text response"
    assert response_log["status_code"] == 200
