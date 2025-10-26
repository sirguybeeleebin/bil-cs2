import logging
import uuid

from django.forms.models import model_to_dict

from app.models import Map, MLPipeline, MLPipelineMetrics, Player, Prediction, Team

log = logging.getLogger(__name__)


class MapRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            obj, _ = Map.objects.update_or_create(
                map_id=data.get("map_id"), defaults=data
            )
            return model_to_dict(obj)
        except Exception:
            log.exception("Ошибка при upsert Map")
            return None


class TeamRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            obj, _ = Team.objects.update_or_create(
                team_id=data.get("team_id"), defaults=data
            )
            return model_to_dict(obj)
        except Exception:
            log.exception("Ошибка при upsert Team")
            return None


class PlayerRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            obj, _ = Player.objects.update_or_create(
                player_id=data.get("player_id"), defaults=data
            )
            return model_to_dict(obj)
        except Exception:
            log.exception("Ошибка при upsert Player")
            return None


class MLPipelineRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            obj, _ = MLPipeline.objects.update_or_create(
                pipeline_id=data.get("pipeline_id"), defaults=data
            )
            return model_to_dict(obj)
        except Exception:
            log.exception("Ошибка при upsert MLPipeline")
            return None


class MLPipelineMetricsRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            obj, _ = MLPipelineMetrics.objects.update_or_create(
                metrics_id=data.get("metrics_id"), defaults=data
            )
            return model_to_dict(obj)
        except Exception:
            log.exception("Ошибка при upsert MLPipelineMetrics")
            return None


class PredictionRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            team1_players = data.pop("team1_players", None)
            team2_players = data.pop("team2_players", None)

            obj, _ = Prediction.objects.update_or_create(
                prediction_id=data.get("prediction_id"), defaults=data
            )
            if team1_players is not None:
                obj.team1_players.set(team1_players)
            if team2_players is not None:
                obj.team2_players.set(team2_players)

            return model_to_dict(obj)
        except Exception:
            log.exception("Ошибка при upsert Prediction")
            return None

    def get_by_id(self, prediction_id: uuid.UUID) -> dict | None:
        try:
            obj = Prediction.objects.filter(prediction_id=prediction_id).first()
            return model_to_dict(obj) if obj else None
        except Exception:
            log.exception("Ошибка при get_by_id Prediction")
            return None
