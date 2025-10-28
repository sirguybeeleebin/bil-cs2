from typing import Type

from rest_framework import permissions, serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView, Request

from backend.repositories import MapRepository, PlayerRepository, TeamRepository


def make_map_search_handler(repo: MapRepository) -> Type[APIView]:
    class MapSearchHandler(APIView):
        permission_classes = [permissions.IsAuthenticated]

        def get(self, request: Request) -> Response:
            name: str = request.query_params.get("name", "")
            try:
                limit: int = int(request.query_params.get("limit", 10))
            except ValueError:
                limit = 10
            try:
                offset: int = int(request.query_params.get("offset", 0))
            except ValueError:
                offset = 0

            results: list[dict] = repo.search_by_name(
                name=name, limit=limit, offset=offset
            )
            return Response(results, status=status.HTTP_200_OK)

    return MapSearchHandler


def make_team_search_handler(repo: TeamRepository) -> Type[APIView]:
    class TeamSearchHandler(APIView):
        permission_classes = [permissions.IsAuthenticated]

        def get(self, request: Request) -> Response:
            name: str = request.query_params.get("name", "")
            try:
                limit: int = int(request.query_params.get("limit", 10))
            except ValueError:
                limit = 10
            try:
                offset: int = int(request.query_params.get("offset", 0))
            except ValueError:
                offset = 0

            results: list[dict] = repo.search_by_name(
                name=name, limit=limit, offset=offset
            )
            return Response(results, status=status.HTTP_200_OK)

    return TeamSearchHandler


def make_player_search_handler(repo: PlayerRepository) -> Type[APIView]:
    class PlayerSearchHandler(APIView):
        permission_classes = [permissions.IsAuthenticated]

        def get(self, request: Request) -> Response:
            name: str = request.query_params.get("name", "")
            try:
                limit: int = int(request.query_params.get("limit", 10))
            except ValueError:
                limit = 10
            try:
                offset: int = int(request.query_params.get("offset", 0))
            except ValueError:
                offset = 0

            results: list[dict] = repo.search_by_name(
                name=name, limit=limit, offset=offset
            )
            return Response(results, status=status.HTTP_200_OK)

    return PlayerSearchHandler


class ForecastRequestSerializer(serializers.Serializer):
    map_id = serializers.IntegerField()
    team1_id = serializers.IntegerField()
    team2_id = serializers.IntegerField()
    start_ct_team_id = serializers.IntegerField()
    team1_player1_id = serializers.IntegerField()
    team1_player2_id = serializers.IntegerField()
    team1_player3_id = serializers.IntegerField()
    team1_player4_id = serializers.IntegerField()
    team1_player5_id = serializers.IntegerField()
    team2_player1_id = serializers.IntegerField()
    team2_player2_id = serializers.IntegerField()
    team2_player3_id = serializers.IntegerField()
    team2_player4_id = serializers.IntegerField()
    team2_player5_id = serializers.IntegerField()


class ForecastResponseSerializer(serializers.Serializer):
    team1_id = serializers.IntegerField()
    team2_id = serializers.IntegerField()
    team1_win_probability = serializers.FloatField()
    team2_win_probability = serializers.FloatField()


def make_forecast_handler() -> Type[APIView]:
    class ForecastHandler(APIView):
        permission_classes = [permissions.IsAuthenticated]

        def post(self, request: Request) -> Response:
            serializer = ForecastRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            data = serializer.validated_data

            # result = forecast_service.predict(**data)
            result = {"team1_win_probability": 0.5, "team2_win_probability": 0.5}
            response_data = {
                "team1_id": data["team1_id"],
                "team2_id": data["team2_id"],
                "team1_win_probability": result.get("team1_win_probability", 0.0),
                "team2_win_probability": result.get("team2_win_probability", 0.0),
            }

            response_serializer = ForecastResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)

    return ForecastHandler
