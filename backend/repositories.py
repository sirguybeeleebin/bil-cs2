from datetime import datetime, timezone
from uuid import UUID

from backend.models import Map, MLPipeline, MLPipelineMetric, Player, Team


def to_dict(obj):
    """Преобразует объект Django в dict, исключая _state."""
    d = obj.__dict__.copy()
    d.pop("_state", None)
    # UUID поля приведём к строке
    for k, v in d.items():
        if isinstance(v, UUID):
            d[k] = str(v)
    return d


class MapRepository:
    def upsert(self, data: dict) -> dict | None:
        now = datetime.now(timezone.utc)
        data = data.copy()
        data["updated_at"] = now
        obj, created = Map.objects.update_or_create(
            map_id=data["map_id"], defaults=data
        )
        return to_dict(obj) if created else None


class TeamRepository:
    def upsert(self, data: dict) -> dict | None:
        now = datetime.now(timezone.utc)
        data = data.copy()
        data["updated_at"] = now
        obj, created = Team.objects.update_or_create(
            team_id=data["team_id"], defaults=data
        )
        return to_dict(obj) if created else None


class PlayerRepository:
    def upsert(self, data: dict) -> dict | None:
        now = datetime.now(timezone.utc)
        data = data.copy()
        data["updated_at"] = now
        obj, created = Player.objects.update_or_create(
            player_id=data["player_id"], defaults=data
        )
        return to_dict(obj) if created else None


class MLPipelineRepository:
    def upsert(self, data: dict) -> dict | None:
        obj, created = MLPipeline.objects.update_or_create(
            pipeline_file=data["pipeline_file"], defaults=data
        )
        return to_dict(obj) if created else None


class MLPipelineMetricRepository:
    def upsert(self, data: dict) -> dict | None:
        ml_pipeline = MLPipeline.objects.get(ml_pipeline_id=data["ml_pipeline_id"])
        obj, created = MLPipelineMetric.objects.update_or_create(
            ml_pipeline=ml_pipeline, defaults=data
        )
        return to_dict(obj) if created else None
