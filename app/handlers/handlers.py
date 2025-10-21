import json
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

log = logging.getLogger(__name__)


class GetMapByNameHandler(APIView):
    def __init__(self, cs2_service, **kwargs):
        super().__init__(**kwargs)
        self.cs2_service = cs2_service

    def get(self, request, name: str):
        log.info(f"GET /maps/{name}")
        result = self.cs2_service.get_map_by_name(name)
        if not result:
            return Response({"error": f"Карта '{name}' не найдена"}, status=status.HTTP_404_NOT_FOUND)
        return Response(result, status=status.HTTP_200_OK)


class GetTeamByNameHandler(APIView):
    def __init__(self, cs2_service, **kwargs):
        super().__init__(**kwargs)
        self.cs2_service = cs2_service

    def get(self, request, name: str):
        log.info(f"GET /teams/{name}")
        result = self.cs2_service.get_team_by_name(name)
        if not result:
            return Response({"error": f"Команда '{name}' не найдена"}, status=status.HTTP_404_NOT_FOUND)
        return Response(result, status=status.HTTP_200_OK)


class GetPlayerByNameHandler(APIView):
    def __init__(self, cs2_service, **kwargs):
        super().__init__(**kwargs)
        self.cs2_service = cs2_service

    def get(self, request, name: str):
        log.info(f"GET /players/{name}")
        result = self.cs2_service.get_player_by_name(name)
        if not result:
            return Response({"error": f"Игрок '{name}' не найден"}, status=status.HTTP_404_NOT_FOUND)
        return Response(result, status=status.HTTP_200_OK)


class GetTeam1WinProbabilityHandler(APIView):
    def __init__(self, ml_service, **kwargs):
        super().__init__(**kwargs)
        self.ml_service = ml_service

    def post(self, request):
        try:
            team1 = request.data.get("team1")
            team2 = request.data.get("team2")
        except Exception:
            return Response({"error": "Некорректный JSON"}, status=status.HTTP_400_BAD_REQUEST)

        if not team1 or not team2:
            return Response({"error": "Поля 'team1' и 'team2' обязательны"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            probability = self.ml_service.get_team1_win_probability(team1, team2)
            return Response({
                "team1": team1,
                "team2": team2,
                "probability": probability,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            log.error(f"Ошибка при расчете вероятности: {e}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
