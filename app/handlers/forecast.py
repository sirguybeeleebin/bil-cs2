from __future__ import annotations

from typing import Any, Callable, Dict, Type

from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.repositories.ml_forecast import MLForecastRepository


class ForecastRequestSerializer(serializers.Serializer):
    map_id = serializers.UUIDField()
    team1_id = serializers.UUIDField()
    team2_id = serializers.UUIDField()
    start_ct_team_id = serializers.UUIDField()

    team1_player1_id = serializers.UUIDField()
    team1_player2_id = serializers.UUIDField()
    team1_player3_id = serializers.UUIDField()
    team1_player4_id = serializers.UUIDField()
    team1_player5_id = serializers.UUIDField()

    team2_player1_id = serializers.UUIDField()
    team2_player2_id = serializers.UUIDField()
    team2_player3_id = serializers.UUIDField()
    team2_player4_id = serializers.UUIDField()
    team2_player5_id = serializers.UUIDField()

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data["team1_id"] == data["team2_id"]:
            raise serializers.ValidationError("Команды должны быть разными")
        return data


class ForecastResponseSerializer(serializers.Serializer):
    ml_forecast_id = serializers.UUIDField()
    status = serializers.CharField()
    team1_win_probability = serializers.FloatField(required=False)
    team2_win_probability = serializers.FloatField(required=False)
    created_at = serializers.DateTimeField(required=False)


def make_forecast_handler(
    ml_forecast_repository: MLForecastRepository,
    run_inference_task: Callable[[str], Any],
) -> Type[APIView]:
    class ForecastHandler(APIView):
        permission_classes = [IsAuthenticated]

        def post(self, request, *args: Any, **kwargs: Any) -> Response:
            request_serializer = ForecastRequestSerializer(data=request.data)
            request_serializer.is_valid(raise_exception=True)
            data: Dict[str, Any] = request_serializer.validated_data
            ml_forecast: Dict[str, Any] = ml_forecast_repository.upsert(data)
            run_inference_task.delay(str(ml_forecast["ml_forecast_id"]))
            response_payload: Dict[str, Any] = {
                "ml_forecast_id": ml_forecast["ml_forecast_id"],
                "status": "inference_started",
                "team1_win_probability": ml_forecast.get("team1_win_probability"),
                "team2_win_probability": ml_forecast.get("team2_win_probability"),
                "created_at": ml_forecast.get("created_at"),
            }
            response_serializer = ForecastResponseSerializer(response_payload)
            return Response(response_serializer.data)

    return ForecastHandler
