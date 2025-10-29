from datetime import datetime, timezone
from uuid import UUID

from backend.models import Map, MLPipeline, MLPipelineMetric, Player, Team


class MapRepository:
    def upsert(self, map_id: int, name: str) -> dict:
        now = datetime.now(timezone.utc)
        obj, _ = Map.objects.update_or_create(
            map_id=map_id,
            defaults={"name": name, "updated_at": now},
        )
        return obj.__dict__

    def search_by_name(self, name: str, limit: int = 10, offset: int = 0) -> list[dict]:
        queryset = Map.objects.filter(name__icontains=name)
        paginated_qs = queryset[offset : offset + limit]
        return [obj.__dict__ for obj in paginated_qs]


class TeamRepository:
    def upsert(self, team_id: int, name: str) -> dict:
        now = datetime.now(timezone.utc)
        obj, _ = Team.objects.update_or_create(
            team_id=team_id,
            defaults={"name": name, "updated_at": now},
        )
        return obj.__dict__

    def search_by_name(self, name: str, limit: int = 10, offset: int = 0) -> list[dict]:
        queryset = Team.objects.filter(name__icontains=name)
        paginated_qs = queryset[offset : offset + limit]
        return [obj.__dict__ for obj in paginated_qs]


class PlayerRepository:
    def upsert(self, player_id: int, name: str) -> dict:
        now = datetime.now(timezone.utc)
        obj, _ = Player.objects.update_or_create(
            player_id=player_id,
            defaults={"name": name, "updated_at": now},
        )
        return obj.__dict__

    def search_by_name(self, name: str, limit: int = 10, offset: int = 0) -> list[dict]:
        queryset = Player.objects.filter(name__icontains=name)
        paginated_qs = queryset[offset : offset + limit]
        return [obj.__dict__ for obj in paginated_qs]


class MLPipelineRepository:
    def upsert(
        self,
        ml_pipeline_id: UUID,
        pipeline_file_path: str,
        metrics_file_path: str,
    ) -> dict:
        obj, _ = MLPipeline.objects.update_or_create(
            ml_pipeline_id=ml_pipeline_id,
            defaults={
                "pipeline_file_path": pipeline_file_path,
                "metrics_file_path": metrics_file_path,
            },
        )
        return obj.__dict__

    def search_by_file(
        self, pipeline_file_path: str = None, metrics_file_path: str = None
    ) -> list[dict]:
        queryset = MLPipeline.objects.all()
        if pipeline_file_path:
            queryset = queryset.filter(pipeline_file_path__icontains=pipeline_file_path)
        if metrics_file_path:
            queryset = queryset.filter(metrics_file_path__icontains=metrics_file_path)
        return [obj.__dict__ for obj in queryset]


class MLPipelineMetricRepository:
    def upsert(
        self,
        ml_pipeline_metric_id: UUID,
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
    ) -> dict:
        ml_pipeline = MLPipeline.objects.get(ml_pipeline_id=ml_pipeline_id)
        obj, _ = MLPipelineMetric.objects.update_or_create(
            ml_pipeline_metric_id=ml_pipeline_metric_id,
            defaults={
                "ml_pipeline": ml_pipeline,
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
        return obj.__dict__
