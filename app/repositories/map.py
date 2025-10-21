import logging
from internal.models import Map

log = logging.getLogger(__name__)

class MapRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            map_id = data.pop("map_id")
            map_obj, created = Map.objects.update_or_create(
                map_id=map_id,
                defaults=data
            )
            action = "создан" if created else "обновлён"
            log.info(f"Map {action}: {map_obj.__dict__}")
            return map_obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при upsert Map: {data}, {e}")
            return None

    def get_by_name(self, name: str) -> dict | None:
        try:
            obj = Map.objects.filter(name__iexact=name).first()
            if not obj:
                log.info(f"Карта с именем '{name}' не найдена")
                return None
            log.info(f"Найдена карта: {obj.__dict__}")
            return obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при поиске карты '{name}': {e}")
            return None

def make_map_repository() -> MapRepository:
    return MapRepository()
