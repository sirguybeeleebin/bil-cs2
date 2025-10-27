from typing import Any, Dict, List, Type

from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.repositories.player import PlayerRepository


class PlayerResponseSerializer(serializers.Serializer):
    player_id = serializers.UUIDField()
    name = serializers.CharField()
    team_id = serializers.UUIDField()
    created_at = serializers.DateTimeField(required=False)


def make_player_get_by_name_handler(
    player_repository: PlayerRepository,
) -> Type[APIView]:
    class PlayerGetByNameHandler(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request, name: str, *args: Any, **kwargs: Any) -> Response:
            player_obj: Dict[str, Any] | None = player_repository.get_by_name(name)
            if player_obj is None:
                return Response({"error": "Player не найден"}, status=404)
            serializer = PlayerResponseSerializer(player_obj)
            return Response(serializer.data)

    return PlayerGetByNameHandler


def make_player_search_by_name_handler(
    player_repository: PlayerRepository,
) -> Type[APIView]:
    class PlayerSearchByNameHandler(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request, name: str, *args: Any, **kwargs: Any) -> Response:
            players: List[Dict[str, Any]] = player_repository.search_by_name(name)
            serializer = PlayerResponseSerializer(players, many=True)
            return Response(serializer.data)

    return PlayerSearchByNameHandler
