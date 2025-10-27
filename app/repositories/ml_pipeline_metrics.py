import logging
import uuid

from app.models.ml_pipeline_metrics import MLPipelineMetrics

log = logging.getLogger(__name__)


class MLPipelineMetricsRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            data["metrics_id"] = data.get("metrics_id") or uuid.uuid4()
            obj, _ = MLPipelineMetrics.objects.update_or_create(
                metrics_id=data["metrics_id"], defaults=data
            )
            return obj.__dict__
        except Exception:
            log.exception("Ошибка при upsert MLPipelineMetrics")
            return None


def make_ml_pipeline_metrics_repository() -> MLPipelineMetricsRepository:
    return MLPipelineMetricsRepository()
