import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from middlewares.logging import LoggingMiddleware


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return JSONResponse({"hello": "world"})

    return app


def test_middleware_adds_request_id(app):
    log_handler = LogCaptureHandler()
    logger = logging.getLogger("auth_service")
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)

    client = TestClient(app)
    response = client.get("/test")

    # Проверяем, что X-Request-ID добавлен
    assert "X-Request-ID" in response.headers
    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 36  # UUID

    # Проверяем, что лог был записан
    assert len(log_handler.records) == 1
    log_record = log_handler.records[0].getMessage()
    log_json = json.loads(log_record)

    # Проверяем основные поля
    assert log_json["request_id"] == request_id
    assert log_json["method"] == "GET"
    assert log_json["path"] == "/test"
    assert log_json["status_code"] == 200
    assert "duration_ms" in log_json
    assert log_json["request_body"] is None
    assert log_json["response_body"] is not None
