import logging
from typing import Any, Callable

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import BaseChannelLayer

from app.repositories.ml_forecast import MLForecastRepository

log = logging.getLogger(__name__)


def make_ml_inference_task(
    ml_forecast_repository: MLForecastRepository,
    channel_layer: BaseChannelLayer,
) -> Callable[[str], dict[str, Any]]:
    @shared_task
    def ml_inference(forecast_id: str) -> dict[str, Any]:
        try:
            forecast = ml_forecast_repository.get_by_id(forecast_id)
            if not forecast:
                log.error(f"Прогноз с id {forecast_id} не найден")
                return {"status": "ошибка", "message": "Прогноз не найден"}

            async_to_sync(channel_layer.group_send)(
                "predictions",
                {
                    "type": "send_prediction",
                    "prediction_id": forecast_id,
                    "team1_id": forecast.get("team1_id"),
                    "team2_id": forecast.get("team2_id"),
                    "team1_win_probability": forecast.get("team1_win_probability"),
                    "team2_win_probability": forecast.get("team2_win_probability"),
                },
            )

            log.info(f"Прогноз {forecast_id} отправлен в WebSocket")
            return {"status": "готово"}

        except Exception as e:
            log.exception(f"Ошибка при выполнении прогноза {forecast_id}: {e}")
            return {"status": "ошибка", "message": str(e)}

    return ml_inference
