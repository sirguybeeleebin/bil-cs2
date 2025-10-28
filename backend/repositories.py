from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from backend.models import Map, MLPipeline, MLPipelineMetric, Player, Team


def to_dict(obj: Map | Team | Player | MLPipeline | MLPipelineMetric) -> dict:
    d = obj.__dict__.copy()
    d.pop("_state", None)
    for k, v in d.items():
        if isinstance(v, UUID):
            d[k] = str(v)
        elif isinstance(v, Path):
            d[k] = str(v)
    return d


class MapRepository:
    def upsert(self, map_id: int, name: str) -> dict | None:
        now = datetime.now(timezone.utc)
        obj, created = Map.objects.update_or_create(
            map_id=map_id, defaults={"name": name, "updated_at": now}
        )
        return to_dict(obj) if created else None

    def search_by_name(self, name: str, limit: int = 10, offset: int = 0) -> list[dict]:
        queryset = Map.objects.filter(name__icontains=name)
        paginated_qs = queryset[offset : offset + limit]
        return [to_dict(obj) for obj in paginated_qs]


class TeamRepository:
    def upsert(self, team_id: int, name: str) -> dict | None:
        now = datetime.now(timezone.utc)
        obj, created = Team.objects.update_or_create(
            team_id=team_id, defaults={"name": name, "updated_at": now}
        )
        return to_dict(obj) if created else None

    def search_by_name(self, name: str, limit: int = 10, offset: int = 0) -> list[dict]:
        queryset = Team.objects.filter(name__icontains=name)
        paginated_qs = queryset[offset : offset + limit]
        return [to_dict(obj) for obj in paginated_qs]


class PlayerRepository:
    def upsert(self, player_id: int, name: str) -> dict | None:
        now = datetime.now(timezone.utc)
        obj, created = Player.objects.update_or_create(
            player_id=player_id, defaults={"name": name, "updated_at": now}
        )
        return to_dict(obj) if created else None

    def search_by_name(self, name: str, limit: int = 10, offset: int = 0) -> list[dict]:
        queryset = Player.objects.filter(name__icontains=name)
        paginated_qs = queryset[offset : offset + limit]
        return [to_dict(obj) for obj in paginated_qs]


class MLPipelineRepository:
    def upsert(self, pipeline_file: Path, metrics_file: Path) -> dict | None:
        obj, created = MLPipeline.objects.update_or_create(
            pipeline_file=pipeline_file, defaults={"metrics_file": metrics_file}
        )
        return to_dict(obj) if created else None

    def get_latest(self) -> dict | None:
        obj = MLPipeline.objects.order_by("-updated_at").first()
        return to_dict(obj) if obj else None


class MLPipelineMetricRepository:
    def upsert(
        self,
        ml_pipeline_id: UUID,
        roc_auc: float,
        f1: float,
        precision: float,
        recall: float,
        accuracy: float,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
    ) -> dict | None:
        ml_pipeline = MLPipeline.objects.get(ml_pipeline_id=ml_pipeline_id)
        obj, created = MLPipelineMetric.objects.update_or_create(
            ml_pipeline=ml_pipeline,
            defaults={
                "roc_auc": roc_auc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            },
        )
        return to_dict(obj) if created else None
