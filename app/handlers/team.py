from typing import Any, Dict, List, Type

from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.repositories.team import TeamRepository


class TeamResponseSerializer(serializers.Serializer):
    team_id = serializers.UUIDField()
    name = serializers.CharField()
    created_at = serializers.DateTimeField(required=False)


def make_team_get_by_name_handler(team_repository: TeamRepository) -> Type[APIView]:
    class TeamGetByNameHandler(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request, name: str, *args: Any, **kwargs: Any) -> Response:
            team_obj: Dict[str, Any] | None = team_repository.get_by_name(name)
            if team_obj is None:
                return Response({"error": "Team не найден"}, status=404)
            serializer = TeamResponseSerializer(team_obj)
            return Response(serializer.data)

    return TeamGetByNameHandler


def make_team_search_by_name_handler(team_repository: TeamRepository) -> Type[APIView]:
    class TeamSearchByNameHandler(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request, name: str, *args: Any, **kwargs: Any) -> Response:
            teams: List[Dict[str, Any]] = team_repository.search_by_name(name)
            serializer = TeamResponseSerializer(teams, many=True)
            return Response(serializer.data)

    return TeamSearchByNameHandler
