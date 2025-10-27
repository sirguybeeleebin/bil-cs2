from __future__ import annotations

from typing import Any, Dict, List, Type

from rest_framework import serializers
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.repositories.map import MapRepository


class MapResponseSerializer(serializers.Serializer):
    map_id = serializers.UUIDField()
    name = serializers.CharField()
    created_at = serializers.DateTimeField(required=False)


def make_map_get_by_name_handler(map_repository: MapRepository) -> Type[APIView]:
    class MapGetByNameHandler(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request, name: str, *args: Any, **kwargs: Any) -> Response:
            map_obj: Dict[str, Any] | None = map_repository.get_by_name(name)
            if map_obj is None:
                return Response({"error": "Map не найден"}, status=404)
            serializer = MapResponseSerializer(map_obj)
            return Response(serializer.data)

    return MapGetByNameHandler


def make_map_search_by_name_handler(map_repository: MapRepository) -> Type[APIView]:
    class MapSearchByNameHandler(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request, name: str, *args: Any, **kwargs: Any) -> Response:
            maps: List[Dict[str, Any]] = map_repository.search_by_name(name)
            serializer = MapResponseSerializer(maps, many=True)
            return Response(serializer.data)

    return MapSearchByNameHandler
