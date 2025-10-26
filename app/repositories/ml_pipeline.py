import logging
import uuid

from app.models.ml_pipeline import MLPipeline

log = logging.getLogger(__name__)


class MLPipelineRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            data["pipeline_id"] = data.get("pipeline_id") or uuid.uuid4()
            obj, _ = MLPipeline.objects.update_or_create(
                pipeline_id=data["pipeline_id"], defaults=data
            )
            return obj.__dict__
        except Exception:
            log.exception("Ошибка при upsert MLPipeline")
            return None


def make_ml_pipeline_repository() -> MLPipelineRepository:
    return MLPipelineRepository()
