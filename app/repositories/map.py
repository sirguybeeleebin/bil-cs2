import logging
import uuid

from app.models.map import Map

log = logging.getLogger(__name__)


class MapRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            data["map_id"] = data.get("map_id") or uuid.uuid4()
            obj, _ = Map.objects.update_or_create(map_id=data["map_id"], defaults=data)
            return obj.__dict__
        except Exception:
            log.exception("Ошибка при upsert Map")
            return None

    def search_by_name(self, name: str) -> list[dict]:
        try:
            results = Map.objects.filter(name__icontains=name)
            return [obj.__dict__ for obj in results]
        except Exception:
            log.exception("Ошибка при search_by_name Map")
            return []

    def get_by_name(self, name: str) -> dict | None:
        try:
            obj = Map.objects.get(name=name)
            return obj.__dict__
        except Map.DoesNotExist:
            return None
        except Exception:
            log.exception("Ошибка при get_by_name Map")
            return None


def make_map_repository() -> MapRepository:
    return MapRepository()
