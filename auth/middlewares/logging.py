import json
import logging
import time
import uuid
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("auth_service")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class JsonLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()

        try:
            body_bytes = await request.body()
            request_body = body_bytes.decode("utf-8") if body_bytes else None
        except Exception:
            request_body = None

        response = await call_next(request)

        response_body = None

        if hasattr(response, "body_iterator"):
            body_chunks = []
            async for chunk in response.body_iterator:
                body_chunks.append(chunk)
            response_body_bytes = b"".join(body_chunks)

            async def async_iterator(data: bytes):
                yield data

            response.body_iterator = async_iterator(response_body_bytes)

            try:
                response_body = response_body_bytes.decode("utf-8")
                try:
                    response_body = json.loads(response_body)
                except Exception:
                    pass
            except Exception:
                response_body = str(response_body_bytes)

        elif hasattr(response, "body"):
            response_body_bytes = response.body
            response_body = (
                response_body_bytes.decode("utf-8") if response_body_bytes else None
            )

        duration_ms = (time.time() - start_time) * 1000

        log_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "request_body": request_body,
            "response_body": response_body,
        }

        logger.info(json.dumps(log_data))

        response.headers["X-Request-ID"] = request_id
        return response
