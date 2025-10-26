import logging
import json
import uuid
from django.conf import settings  # <--- исправлено
from app.tasks import send_log_to_fluentd

logger = logging.getLogger("django.request_json")

class JsonLoggingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.fluentd_url = settings.FLUENTD_URL  # <-- берем из settings
        self.send_task = send_log_to_fluentd

    def __call__(self, request):
        request_id = uuid.uuid4().hex
        request.request_id = request_id

        try:
            raw_body = request.body.decode("utf-8")
            try:
                body = json.loads(raw_body)
            except Exception:
                body = raw_body
        except Exception:
            body = "<cannot decode>"

        request_log = {
            "request_id": request_id,
            "type": "request",
            "method": request.method,
            "path": request.get_full_path(),
            "body": body,
            "headers": dict(request.headers),
        }

        logger.info(json.dumps(request_log, ensure_ascii=False))
        self.send_task.delay(self.fluentd_url, request_log)

        response = self.get_response(request)

        try:
            raw_content = response.content.decode("utf-8")
            try:
                content = json.loads(raw_content)
            except Exception:
                content = raw_content
        except Exception:
            content = "<cannot decode>"

        response_log = {
            "request_id": request_id,
            "type": "response",
            "method": request.method,
            "path": request.get_full_path(),
            "status_code": response.status_code,
            "body": content,
        }

        logger.info(json.dumps(response_log, ensure_ascii=False))
        self.send_task.delay(self.fluentd_url, response_log)

        return response
