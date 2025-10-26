import logging
import uuid

from app.models.ml_forecast import MLForecast

log = logging.getLogger(__name__)


class MLForecastRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            team1_players = data.pop("team1_players", [])
            team2_players = data.pop("team2_players", [])

            data["ml_forecast_id"] = data.get("ml_forecast_id") or uuid.uuid4()
            obj, _ = MLForecast.objects.update_or_create(
                ml_forecast_id=data["ml_forecast_id"], defaults=data
            )

            if team1_players:
                obj.team1_players.set(team1_players)
            if team2_players:
                obj.team2_players.set(team2_players)

            return obj.__dict__
        except Exception:
            log.exception("Ошибка при upsert MLForecast")
            return None

    def get_by_id(self, ml_forecast_id: uuid.UUID) -> dict | None:
        try:
            obj = MLForecast.objects.filter(ml_forecast_id=ml_forecast_id).first()
            return obj.__dict__ if obj else None
        except Exception:
            log.exception("Ошибка при get_by_id MLForecast")
            return None


def make_ml_forecast_repository() -> MLForecastRepository:
    return MLForecastRepository()
