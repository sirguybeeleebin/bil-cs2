from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

from backend.models import Map, MLPipeline, MLPipelineMetric, Player, Team


def to_dict(obj: Union[Map, Team, Player, MLPipeline, MLPipelineMetric]) -> dict:
    """Преобразует объект Django в dict, исключая _state."""
    d = obj.__dict__.copy()
    d.pop("_state", None)
    for k, v in d.items():
        if isinstance(v, UUID):
            d[k] = str(v)
        elif isinstance(v, Path):
            d[k] = str(v)
    return d


class MapRepository:
    def upsert(self, map_id: int, name: str) -> Optional[dict]:
        now: datetime = datetime.now(timezone.utc)
        obj, created = Map.objects.update_or_create(
            map_id=map_id, defaults={"name": name, "updated_at": now}
        )
        return to_dict(obj) if created else None


class TeamRepository:
    def upsert(self, team_id: int, name: str) -> Optional[dict]:
        now: datetime = datetime.now(timezone.utc)
        obj, created = Team.objects.update_or_create(
            team_id=team_id, defaults={"name": name, "updated_at": now}
        )
        return to_dict(obj) if created else None


class PlayerRepository:
    def upsert(self, player_id: int, name: str) -> Optional[dict]:
        now: datetime = datetime.now(timezone.utc)
        obj, created = Player.objects.update_or_create(
            player_id=player_id, defaults={"name": name, "updated_at": now}
        )
        return to_dict(obj) if created else None


class MLPipelineRepository:
    def upsert(self, pipeline_file: Path, metrics_file: Path) -> Optional[dict]:
        obj, created = MLPipeline.objects.update_or_create(
            pipeline_file=pipeline_file, defaults={"metrics_file": metrics_file}
        )
        return to_dict(obj) if created else None


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
    ) -> Optional[dict]:
        ml_pipeline: MLPipeline = MLPipeline.objects.get(ml_pipeline_id=ml_pipeline_id)
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
